"""
MAT with lightweight event-to-image tracking before encoder stages.
This variant keeps MAT's event gating but aligns image features using
InlineTrackerBlock prior to Transformer encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class MSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MSA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        q = q * mask
        k = k * mask
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class MAA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MAA, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.temperature_e = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, me):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        in_me = -1 * (me - 1)
        q_enhe = me * self.temperature_e + in_me
        q = q * q_enhe
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class CMIG(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(CMIG, self).__init__()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, ev):
        x = F.gelu(ev) * x
        x = self.project_out(x)
        return x


class TransformerBlock_MAT(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, use_evs=False, evs_enc=False):
        super(TransformerBlock_MAT, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if not evs_enc:
            if not use_evs:
                self.attn = Attention(dim, num_heads, bias)
            else:
                self.attn = MAA(dim, num_heads, bias)
        else:
            self.attn = MSA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.use_evs = use_evs
        if use_evs:
            self.fusion = CMIG(dim, ffn_expansion_factor, bias)
            self.norm0 = LayerNorm(dim, LayerNorm_type)
            self.norm_evs = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask_e=None):
        if self.use_evs:
            c = x.shape[1]
            events = x[:, c // 2 :, :, :]
            x = x[:, : c // 2, :, :]
            x = self.fusion(self.norm0(x), self.norm_evs(events)) + x
        if mask_e is None:
            x = x + self.attn(self.norm1(x))
        else:
            x = x + self.attn(self.norm1(x), mask_e)
        x = x + self.ffn(self.norm2(x))
        if self.use_evs:
            return torch.cat([x, events], 1), mask_e
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


def conv_down(in_chn, out_chn, bias=False):
    return nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)


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


class UNetEVTransformerBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, num_heads):
        super(UNetEVTransformerBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0, bias=True)
        self.score_predictor = AMMP(out_size, layernorm_type="WithBias")
        self.encoder = [TransformerBlock_MAT(evs_enc=True, dim=out_size, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias") for _ in range(2)]
        self.encoder = nn.Sequential(*self.encoder)
        self.conv_before_merge = nn.Conv2d(out_size, out_size, 1, 1, 0)
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, ratio=None, merge_before_downsample=True):
        b, c, h, w = x.size()
        prev_decision = torch.ones(b, 1, 1, h * w, dtype=x.dtype, device=x.device)
        out = self.conv_1(x)
        for i, enc in enumerate(self.encoder):
            pred_score, weighting = self.score_predictor(out, prev_decision, ratio)
            mask = pred_score
            #  -------------ECSG------------
            out = out*weighting + out
            #  -----------------------------
            out = enc(out, mask) 
            prev_decision = mask
        
        out = out + self.identity(x)
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.conv_before_merge(out_down)
            else:
                out = self.conv_before_merge(out)
            return out_down, out, mask
        out = self.conv_before_merge(out)
        return out, mask


class CustomSequential(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            x1, x2 = x
            x = module(x1, x2)
        return x


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


class MAT_with_tracking(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=[8, 8, 7],
        num_refinement_blocks=2,
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        use_tracking=True,
    ):
        super(MAT_with_tracking, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = CustomSequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**0),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    use_evs=True,
                )
                for _ in range(num_blocks[0])
            ]
        )
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = CustomSequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    use_evs=True,
                )
                for _ in range(num_blocks[1])
            ]
        )
        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = CustomSequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    use_evs=True,
                )
                for _ in range(num_blocks[2])
            ]
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )
        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )
        self.up2_1 = Upsample(int(dim * 2**1))
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2**1), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )
        self.refinement = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_refinement_blocks)
            ]
        )
        self.down_path_ev = nn.ModuleList()
        prev_channels = dim
        depth = len(num_blocks)
        self.depth = depth
        for i in range(depth):
            downsample = True if (i + 1) < depth else False
            if i < self.depth:
                self.down_path_ev.append(UNetEVTransformerBlock(prev_channels, (2**i) * dim, downsample, num_heads=heads[i]))
            prev_channels = (2**i) * dim
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_ev1 = nn.Conv2d(6, dim, 3, 1, 1)
        self.fuse_before_downsample = True
        self.use_tracking = use_tracking
        self.trackers = nn.ModuleList(
            [
                InlineTrackerBlock(dim, lstm_hidden_dim=dim),
                InlineTrackerBlock(dim * 2, lstm_hidden_dim=dim * 2),
                InlineTrackerBlock(dim * 4, lstm_hidden_dim=dim * 4),
            ]
        )

    def forward(self, x, event):
        inp_img = x
        events = event
        ratio = get_non_zero_ratio(events)

        ev = []
        se = []
        e1 = self.conv_ev1(events)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth - 1:
                e1, e1_up, score = down(e1, ratio[i], self.fuse_before_downsample)
                ev.append(e1_up if self.fuse_before_downsample else e1)
            else:
                e1, score = down(e1, ratio[i], self.fuse_before_downsample)
                ev.append(e1)
            se.append(score)

        inp_enc_level1 = self.patch_embed(inp_img)
        if self.use_tracking:
            inp_enc_level1 = self.trackers[0](inp_enc_level1, ev[0])
        inp_enc_level1 = torch.cat([inp_enc_level1, ev[0]], 1)
        out_enc_level1, _ = self.encoder_level1((inp_enc_level1, se[0]))
        out_enc_level1 = out_enc_level1[:, : out_enc_level1.shape[1] // 2, :, :]

        inp_enc_level2 = self.down1_2(out_enc_level1)
        if self.use_tracking:
            inp_enc_level2 = self.trackers[1](inp_enc_level2, ev[1])
        inp_enc_level2 = torch.cat([inp_enc_level2, ev[1]], 1)
        out_enc_level2, _ = self.encoder_level2((inp_enc_level2, se[1]))
        out_enc_level2 = out_enc_level2[:, : out_enc_level2.shape[1] // 2, :, :]

        inp_enc_level3 = self.down2_3(out_enc_level2)
        if self.use_tracking:
            inp_enc_level3 = self.trackers[2](inp_enc_level3, ev[2])
        inp_enc_level3 = torch.cat([inp_enc_level3, ev[2]], 1)
        out_enc_level3, _ = self.encoder_level3((inp_enc_level3, se[2]))
        inp_dec_level3 = out_enc_level3[:, : out_enc_level3.shape[1] // 2, :, :]

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1

    def reset_lstm_states(self):
        for i in range(len(self.trackers)):
            self.trackers[i].reset_states()

