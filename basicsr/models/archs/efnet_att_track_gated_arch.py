"""
EFNet_att_track_fusion_new with event-gated skip connections in stage 2.
This variant uses CMIG-style gating to modulate skip features by events.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

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


class CMIG(nn.Module):
    def __init__(self, dim, bias=False):
        super(CMIG, self).__init__()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, ev):
        x = F.gelu(ev) * x
        x = self.project_out(x)
        return x


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
            input_dim=lstm_hidden_dim,
            hidden_dim=lstm_hidden_dim,
            kernel_size=3,
            bias=True,
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


class EFNet_att_track_fusion_event_gated_skip(nn.Module):
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
        super(EFNet_att_track_fusion_event_gated_skip, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.use_tracking = use_tracking
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)
        self.event_gates = nn.ModuleList()

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
                    use_tracking=False,
                )
            )
            if i < self.depth:
                self.down_path_ev.append(
                    UNetEVConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope)
                )
                if i < self.depth - 1:
                    self.event_gates.append(CMIG((2**i) * wf, bias=False))
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

    def forward(self, x, event, mask=None):
        image = x

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
                gated_enc = self.event_gates[i](encs[i], ev_features[i])
                gated_dec = self.event_gates[i](decs[-i - 1], ev_features[i])
                if mask is not None:
                    x2, x2_up = down(
                        x2,
                        gated_enc,
                        gated_dec,
                        mask=masks[i],
                        event_feat=ev_features[i] if self.use_tracking else None,
                    )
                else:
                    x2, x2_up = down(
                        x2,
                        gated_enc,
                        gated_dec,
                        event_feat=ev_features[i] if self.use_tracking else None,
                    )
                blocks.append(self.event_gates[i](x2_up, ev_features[i]))
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

