
import torch
import torch.nn as nn
import math
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock, LayerNorm2d
from basicsr.models.archs.EFNet_arch import UNetUpBlock, UNetConvBlock, UNetEVConvBlock
from torch.nn import functional as F

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Supervised Attention Module
## https://github.com/swz30/MPRNet
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
        x1 = x1*x2
        x1 = x1+x
        return x1, img
class SimpleGate(nn.Module):
    def forward(self, x):
        # 将通道分成两半，要求 C 必须为偶数（本网络设计中通过 Expand 保证）
        x1, x2 = x.chunk(2, dim=1) # 平分成两份
        return x1 * x2

class NAFBlock(nn.Module):

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        # ...existing code...
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # Depthwise Conv：逐通道卷积，计算量低，用于局部空间信息混合
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        # ...existing code...
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention (SCA)
        # 先对空间做全局均值池化得到每通道统计，再用 1x1 生成通道权重进行调制
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate：通过通道二分相乘引入“隐式非线性”
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        # FFN-like：用 1x1 扩张/压缩通道实现逐位置的通道混合
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # LayerNorm2d：对每个像素位置做通道归一化（复原任务常用，稳定训练）
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout：可选正则（复原任务中常设为 0）
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # 残差缩放（初始化为 0）：训练初期更像恒等映射，梯度更稳定
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # 主分支输入缓存，用于残差连接
        x = inp

        # 预归一化：先 LN 再做卷积变换（类似 Pre-LN 的稳定性思路）
        x = self.norm1(x)

        # 1x1 扩张通道 -> DWConv 空间混合 -> 门控
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        # SCA：用全局统计生成通道权重，对 x 做逐通道调制
        x = x * self.sca(x)

        # 1x1 压回原通道数
        x = self.conv3(x)

        x = self.dropout1(x)

        # 第一次残差：y = inp + beta * F(inp)
        y = inp + x * self.beta

        # FFN-like 分支：LN -> 1x1 扩张 -> 门控 -> 1x1 压回
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        # 第二次残差：输出 = y + gamma * G(y)
        return y + x * self.gamma

class evNAFNet(nn.Module): # wf is width factor
    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4], naf_blocks=[2, 2, 4]):
        super(evNAFNet, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.naf_blocks = naf_blocks

        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()

        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        # event
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False 
            # stage 1 保持原样，stage 2用nafblock
            self.down_path_1.append(
                UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, num_heads=self.num_heads[i])
            )
            self.down_path_2.append(NAFConvBlock(prev_channels, (2**i) * wf, downsample, use_emgc=downsample, naf_blocks=self.naf_blocks[i]))
            # ev encoder
            if i < self.depth:
                self.down_path_ev.append(
                    UNetEVConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope)
                )

            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.up_path_2.append(NAFUpBlock(prev_channels, (2**i)*wf, naf_blocks=self.naf_blocks[i]))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, event, mask=None):
        image = x

        ev = []
        #EVencoder
        e1 = self.conv_ev1(event)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth-1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)

        #stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:

                x1, x1_up = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
            else:
                x1 = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)


        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)

        #stage 2
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                if mask is not None:
                    x2, x2_up = down(x2, encs[i], decs[-i-1], mask=masks[i])
                else:
                    x2, x2_up = down(x2, encs[i], decs[-i-1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]))

        out_2 = self.last(x2)
        out_2 = out_2 + image

        return [out_1, out_2]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class NAFConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, use_emgc=False, num_heads=None,
                 naf_blocks=1, naf_ffn_expansion=2, naf_dw_kernel=3, naf_dropout=0.0):
        super(NAFConvBlock, self).__init__()
        self.downsample = downsample
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        self.conv_1 = nn.Identity() if in_size == out_size else nn.Conv2d(in_size, out_size, 1, 1, 0)     
        self.naf_blocks = nn.Sequential(*[
            NAFBlock(out_size, DW_Expand=naf_dw_kernel, FFN_Expand=naf_ffn_expansion, drop_out_rate=naf_dropout)
            for _ in range(naf_blocks)
        ])
        
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)
        out = self.naf_blocks(out)

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1-mask)*enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask*dec)
            out = out + out_enc + out_dec        
            
        if event_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            out = self.image_event_transformer(out, event_filter) 
             
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter) 

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)
                print("warning: no downsample but merge after")
        
class NAFEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, naf_blocks=1, 
                 naf_ffn_expansion=2, naf_dw_kernel=3, naf_dropout=0.0):
        super(NAFEVConvBlock, self).__init__()
        self.downsample = downsample
        self.conv_1 = nn.Identity() if in_size == out_size else nn.Conv2d(in_size, out_size, 1, 1, 0)
        # naf blocks
        self.naf_blocks = nn.Sequential(*[
            NAFBlock(out_size, DW_Expand=naf_dw_kernel, FFN_Expand=naf_ffn_expansion, drop_out_rate=naf_dropout)
            for _ in range(naf_blocks)
        ])
        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0)
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)
    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)
        out = self.naf_blocks(out)
             
        if self.downsample:

            out_down = self.downsample(out)
            
            if not merge_before_downsample: 
            
                out_down = self.conv_before_merge(out_down)
            else : 
                out = self.conv_before_merge(out)
            return out_down, out # out_down 本分支，out拿给另一个branch

        else: # bottleneck
            out = self.conv_before_merge(out)
            return out
        
class NAFUpBlock(nn.Module):

    def __init__(self, in_size, out_size, naf_blocks=1):
        super(NAFUpBlock, self).__init__()
        assert in_size == out_size * 2, "in_size must be 2 times of out_size"
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = NAFConvBlock(in_size, out_size, False, naf_blocks=naf_blocks)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

if __name__ == "__main__":
    pass
