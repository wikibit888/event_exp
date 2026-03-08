"""
 - 核心设计：在 EFNet_att_track_fusion 框架中引入 MAT 的事件稀疏门控（AMMP），先筛选/加权事件特征，再做图像-事件融
    合。
  - 主要改动：新增 AMMP 与 EventGateBlock，事件分支每层先门控后送入 EventImage_ChannelAttentionTransformerBlock；保
    持双阶段 UNet 与 [out_1, out_3] 输出不变。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock


def conv3x3(in_chn, out_chn, bias=True):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)


def conv_down(in_chn, out_chn, bias=False):
    return nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def get_non_zero_ratio(x):
    x_down_2 = torch.nn.functional.max_pool2d(x.float(), kernel_size=2, stride=2)
    x_down_4 = torch.nn.functional.max_pool2d(x_down_2, kernel_size=2, stride=2)
    num_nonzero_1 = torch.sum(torch.sum(x != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_2 = torch.sum(torch.sum(x_down_2 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_3 = torch.sum(torch.sum(x_down_4 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)

    result1 = x.shape[0] / x.numel() * num_nonzero_1.float()
    result2 = x.shape[0] / x_down_2.numel() * num_nonzero_2.float()
    result3 = x.shape[0] / x_down_4.numel() * num_nonzero_3.float()
    return [abs(result1), abs(result2), abs(result3)]


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layernorm_type):
        super(LayerNorm, self).__init__()
        if layernorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weights = torch.exp(self.weight)
        return nn.functional.linear(input, positive_weights, self.bias)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AMMP(nn.Module):
    def __init__(self, embed_dim, layernorm_type):
        super().__init__()
        self.in_norm = LayerNorm(embed_dim, layernorm_type)
        self.bins = 6
        self.to_controls = PositiveLinear(self.bins, embed_dim * 2, bias=False)
        self.in_proj = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False), nn.GELU())
        self.conv_compress = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2, bias=False),
            nn.GELU(),
        )
        self.channel_compress = nn.Sequential(
            ChannelPool(), nn.Conv2d(2, 2, kernel_size=1, bias=False), nn.GELU()
        )
        self.in_compress3 = nn.Sequential(nn.Linear(4, 1, bias=False), nn.GELU())
        self.alpha = nn.Parameter(torch.ones(1, 1, 1) * 0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask, ratio):
        b, c, h, w = x.size()
        x = self.in_norm(x)
        local_x = self.in_proj(x)

        ratio = ratio[:, None, None, :]
        scale = self.to_controls(ratio)

        mask = rearrange(mask, "b head c (h w) -> b (head c) h w", head=1, h=h, w=w)
        denom = torch.sum(torch.sum(mask, dim=-1, keepdim=True), dim=-2, keepdim=True)
        global_x = (local_x * mask).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True) / denom
        global_x[global_x == torch.inf] = 0

        x_1 = self.channel_compress(local_x * global_x)
        x_1 = rearrange(x_1, "b (i c) h w -> b i (h w) c", c=2, h=h, w=w)

        x_2 = torch.cat([local_x, global_x.expand(b, c, h, w)], dim=1) * scale.permute(0, -1, 1, 2)
        x_2 = rearrange(x_2, "b (i c) h w -> b i (h w) c", c=2 * c, h=h, w=w)
        x_2 = self.conv_compress(x_2)

        x = self.in_compress3(torch.cat([x_1, x_2], dim=-1))
        output = rearrange(x, "b (head c) h w -> b head c (h w)", head=1)

        self.alpha = ratio.max() if self.alpha < ratio.max() else self.alpha
        m = ratio.max() / self.alpha
        k_ratio = m if m <= 0.4 else 0.4
        k_ratio = torch.where(k_ratio <= 0.004, 0.005, k_ratio)

        indexs = torch.topk(output, k=int(k_ratio * h * w), dim=-1, largest=True, sorted=False)[1]
        new_mask = torch.zeros(b, 1, 1, h * w, device=x.device, requires_grad=False)
        new_mask.scatter_(-1, indexs, 1.0)
        indexs = torch.topk(output, k=int(k_ratio * h * w), dim=-1, largest=False, sorted=False)[1]
        new_mask.scatter_(-1, indexs, 1.0)

        weighting = self.sigmoid(self.alpha) * F.gelu(rearrange(x, "b i (h w) c -> b (i c) h w", c=1, h=h, w=w))
        return new_mask, weighting


class EventGateBlock(nn.Module):
    def __init__(self, dim):
        super(EventGateBlock, self).__init__()
        self.score_predictor = AMMP(dim, layernorm_type="WithBias")

    def forward(self, ev_feat, ratio):
        b, _, h, w = ev_feat.size()
        prev_decision = torch.ones(b, 1, 1, h * w, dtype=ev_feat.dtype, device=ev_feat.device)
        mask, weighting = self.score_predictor(ev_feat, prev_decision, ratio)
        return ev_feat * weighting + ev_feat, mask


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class InlineTrackerBlock(nn.Module):
    def __init__(self, channels, lstm_hidden_dim=64):
        super(InlineTrackerBlock, self).__init__()
        self.flow_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True)
        self.flow_relu = nn.LeakyReLU(0.2, inplace=False)
        self.flow_conv2 = nn.Conv2d(channels, 2, kernel_size=3, padding=1, bias=True)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.feature_encoder = nn.Conv2d(channels * 2 + 2, lstm_hidden_dim, kernel_size=3, padding=1, bias=True)
        self.feature_encoder_act = nn.LeakyReLU(0.2, inplace=False)
        self.conv_lstm = ConvLSTMCell(
            input_dim=lstm_hidden_dim, hidden_dim=lstm_hidden_dim, kernel_size=3, bias=True
        )
        self.flow_decoder = nn.Conv2d(lstm_hidden_dim, 2, kernel_size=3, padding=1, bias=True)
        self.lstm_h = None
        self.lstm_c = None

    def forward(self, img_feat, event_feat):
        b, _, h, w = img_feat.size()
        device = img_feat.device
        concat_feat = torch.cat([img_feat, event_feat], dim=1)
        flow_features = self.flow_conv1(concat_feat)
        flow_features = self.flow_relu(flow_features)
        initial_flow = self.flow_conv2(flow_features)

        if self.lstm_h is None or self.lstm_c is None or self.lstm_h.size(0) != b:
            self.lstm_h = torch.zeros(b, self.lstm_hidden_dim, h, w, device=device)
            self.lstm_c = torch.zeros(b, self.lstm_hidden_dim, h, w, device=device)
            lstm_input_features = torch.cat([concat_feat, initial_flow], dim=1)
            lstm_input_features = self.feature_encoder(lstm_input_features)
            lstm_input_features = self.feature_encoder_act(lstm_input_features)
            self.lstm_h, self.lstm_c = self.conv_lstm(lstm_input_features, (self.lstm_h, self.lstm_c))
            flow_refinement = self.flow_decoder(self.lstm_h)
            flow = initial_flow + flow_refinement
        else:
            flow = initial_flow

        xx = torch.arange(0, w, device=device).view(1, -1).repeat(h, 1).float() / (w - 1) * 2 - 1
        yy = torch.arange(0, h, device=device).view(-1, 1).repeat(1, w).float() / (h - 1) * 2 - 1
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        grid = grid.permute(0, 2, 3, 1)
        flow_x = flow[:, 0, :, :] / ((w - 1) / 2)
        flow_y = flow[:, 1, :, :] / ((h - 1) / 2)
        flow_scaled = torch.stack([flow_x, flow_y], dim=-1)
        grid_flow = grid + flow_scaled
        warped_feat = F.grid_sample(img_feat, grid_flow, mode="bilinear", padding_mode="border", align_corners=True)
        return warped_feat

    def reset_states(self):
        self.lstm_h = None
        self.lstm_c = None


class EFNet_att_track_fusion_ammp(nn.Module):
    def __init__(
        self,
        in_chn=3,
        ev_chn=6,
        wf=64,
        depth=3,
        fuse_before_downsample=True,
        relu_slope=0.2,
        num_heads=[1, 2, 4],
        use_tracking=True,
    ):
        super(EFNet_att_track_fusion_ammp, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.use_tracking = use_tracking
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.down_path_ev = nn.ModuleList()
        self.ev_gates = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)

        prev_channels = wf
        for i in range(depth):
            downsample = True if (i + 1) < depth else False
            layer_use_tracking = use_tracking and i in [0, 1]
            self.down_path_1.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    num_heads=self.num_heads[i],
                    use_tracking=layer_use_tracking,
                )
            )
            self.down_path_2.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    use_emgc=downsample,
                    use_tracking=layer_use_tracking,
                )
            )
            if i < self.depth:
                self.down_path_ev.append(
                    UNetEVConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope)
                )
                self.ev_gates.append(EventGateBlock((2**i) * wf))
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            prev_channels = (2**i) * wf
        self.sam12 = SAM(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
        self.fine_fusion = BidirectionalFrameFusionBlock(channels=wf)
        self.coarse_map = nn.Conv2d(3, wf, kernel_size=1, padding=0)
        self.coarse_unmap = nn.Conv2d(wf, 3, kernel_size=3, padding=1)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def _get_ratios(self, event):
        ratios = get_non_zero_ratio(event)
        if len(ratios) >= self.depth:
            return ratios[: self.depth]
        return ratios + [ratios[-1] for _ in range(self.depth - len(ratios))]

    def forward(self, x, event, mask=None):
        image = x
        ratios = self._get_ratios(event)

        ev = []
        e1 = self.conv_ev1(event)
        ev_features = []
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth - 1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                ev_feat = e1_up if self.fuse_before_downsample else e1
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev_feat = e1
            ev_feat, _ = self.ev_gates[i](ev_feat, ratios[i])
            ev.append(ev_feat)
            ev_features.append(ev_feat)

        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                x1, x1_up = down(
                    x1,
                    event_filter=ev[i],
                    merge_before_downsample=self.fuse_before_downsample,
                    event_feat=ev_features[i] if self.use_tracking else None,
                )
                encs.append(x1_up)
                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor=0.5**i))
            else:
                x1 = down(
                    x1,
                    event_filter=ev[i],
                    merge_before_downsample=self.fuse_before_downsample,
                    event_feat=ev_features[i] if self.use_tracking else None,
                )

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)

        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                if mask is not None:
                    x2, x2_up = down(
                        x2, encs[i], decs[-i - 1], mask=masks[i], event_feat=ev_features[i] if self.use_tracking else None
                    )
                else:
                    x2, x2_up = down(
                        x2, encs[i], decs[-i - 1], event_feat=ev_features[i] if self.use_tracking else None
                    )
                blocks.append(x2_up)
            else:
                x2 = down(x2, event_feat=ev_features[i] if self.use_tracking else None)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]))

        out_2 = self.last(x2)
        out_2 = out_2 + image

        out_2_map = self.coarse_map(out_2)
        _, fused_feat = self.fine_fusion(out_2_map, x2)
        out_3 = self.last(fused_feat)
        out_3 = out_3 + image

        return [out_1, out_3]

    def reset_lstm_states(self):
        for i in range(min(2, self.depth)):
            if hasattr(self.down_path_1[i], "feature_tracker") and self.down_path_1[i].use_tracking:
                self.down_path_1[i].feature_tracker.reset_states()
            if hasattr(self.down_path_2[i], "feature_tracker") and self.down_path_2[i].use_tracking:
                self.down_path_2[i].feature_tracker.reset_states()


class UNetConvBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        downsample,
        relu_slope,
        use_emgc=False,
        num_heads=None,
        use_tracking=False,
    ):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads
        self.use_tracking = use_tracking

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.use_tracking:
            self.feature_tracker = InlineTrackerBlock(out_size, lstm_hidden_dim=out_size)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(
                out_size,
                num_heads=self.num_heads,
                ffn_expansion_factor=4,
                bias=False,
                LayerNorm_type="WithBias",
            )

    def forward(
        self,
        x,
        enc=None,
        dec=None,
        mask=None,
        event_filter=None,
        merge_before_downsample=True,
        event_feat=None,
    ):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.relu_2(out)
        out = out + self.identity(x)

        if enc is not None and dec is not None and self.use_emgc:
            if mask is not None:
                out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1 - mask) * enc)
                out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask * dec)
                out = out + out_enc + out_dec

        if self.num_heads is not None and event_filter is not None:
            if merge_before_downsample:
                out = self.image_event_transformer(out, event_filter)

        if self.use_tracking and event_feat is not None:
            out = self.feature_tracker(out, event_feat)

        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample and self.num_heads is not None and event_filter is not None:
                out_down = self.image_event_transformer(out_down, event_filter)
            return out_down, out
        else:
            if not merge_before_downsample and self.num_heads is not None and event_filter is not None:
                out = self.image_event_transformer(out, event_filter)
            return out


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size, 1, 1, 0)
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)

        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.conv_before_merge(out_down)
            else:
                out = self.conv_before_merge(out)
            return out_down, out
        else:
            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class SimpleGate(nn.Module):
    def forward(self, x):
        c = x.shape[1] // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]
        return x1 * x2


class FusionSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionSubBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        return x


class BidirectionalFrameFusionBlock(nn.Module):
    def __init__(self, channels):
        super(BidirectionalFrameFusionBlock, self).__init__()
        half = channels // 2
        self.forward_block = FusionSubBlock(in_channels=half + channels, out_channels=channels)
        self.backward_block = FusionSubBlock(in_channels=half + channels, out_channels=channels)

    def forward(self, f_i, f_ip1):
        _, c, _, _ = f_i.shape
        half = c // 2
        fa_i = f_i[:, :half, :, :]
        fb_i = f_i[:, half:, :, :]
        forward_in = torch.cat([fa_i, f_ip1], dim=1)
        f_ip1_new = self.forward_block(forward_in)
        backward_in = torch.cat([f_ip1_new, fb_i], dim=1)
        f_i_new = self.backward_block(backward_in)
        return f_i_new, f_ip1_new

