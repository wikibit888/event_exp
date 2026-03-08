'''
EFNet
@inproceedings{sun2022event,
      author = {Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Van Gool, Luc},
      title = {Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year = 2022
}

本文件为 EFNet 的 PyTorch 实现（图像 + 事件（event）融合去模糊）。
整体结构要点：
1) 事件分支（EV encoder）：对 event tensor 逐尺度编码，得到与图像编码器对齐的多尺度 event 特征（ev[i]）。
2) 两阶段图像 U-Net：
   - Stage1：图像编码/解码，同时在编码阶段通过跨模态注意力与事件特征融合，输出一个中间结果 out_1，并通过 SAM 产生用于 Stage2 的引导特征 sam_feature。
   - Stage2：在 Stage1 的编码/解码特征指导下进行细化重建，输出最终结果 out_2。
3) 可选 mask（通常为遮挡/运动区域等）：在 Stage2 编码阶段启用 EMGC（Encoder/Decoder Mask Guided Correction）对特征进行区域化修正。

张量形状约定（默认 NCHW）：
- image/x: [B, 3, H, W]
- event:   [B, ev_chn, H, W]
- 中间特征：通道数随尺度增加，空间尺寸随 downsample 减半。
'''

import torch
import torch.nn as nn
import math
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock
from torch.nn import functional as F

def conv3x3(in_chn, out_chn, bias=True):
    """3x3 卷积（stride=1），保持空间尺寸不变（H,W 不变）。"""
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    """4x4 卷积下采样（stride=2），空间尺寸减半（H,W -> H/2,W/2）。"""
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    """通用卷积：默认 stride=1 且 padding=kernel_size//2，保持空间尺寸不变。"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    """
    Supervised Attention Module（SAM，来自 MPRNet 思想）
    作用：
    - 从特征 x 预测一张残差图像 img（与输入 x_img 相加得到中间重建）
    - 再从 img 生成注意力图 x2，对特征进行逐像素/逐通道的门控增强
    输入：
    - x:     [B, C, H, W]  当前阶段解码后的特征
    - x_img: [B, 3, H, W]  原始输入图像（作为残差连接/监督信号载体）
    输出：
    - x1:  [B, C, H, W]  经注意力增强后的特征（供下一阶段使用）
    - img: [B, 3, H, W]  中间重建结果（stage1 输出）
    """
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        # 提取特征（保持通道数不变）
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        # 从特征预测 RGB 残差/重建分量
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        # 从重建图像生成注意力图（映射回特征通道数）
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        # x1：特征分支
        x1 = self.conv1(x)
        # img：中间重建结果（对输入图像做残差形式的修正）
        img = self.conv2(x) + x_img
        # x2：注意力图（sigmoid 映射到 [0,1]）
        x2 = torch.sigmoid(self.conv3(img))
        # 以重建图像为“监督提示”生成的注意力对特征做门控，再残差回加原特征
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img

class EFNet(nn.Module):
    """
    EFNet 主网络：
    - 输入：图像 image 与事件 event
    - 输出：[out_1, out_2]（stage1 与 stage2 的重建结果）

    关键超参说明：
    - depth：U-Net 尺度层数（depth=3 表示 3 个编码尺度，最后一层不再下采样）
    - wf：base width（第一层通道数）
    - fuse_before_downsample：
        True  -> 在下采样前用 event 特征做融合（更高分辨率上融合）
        False -> 先下采样再融合（更省显存，但融合发生在更低分辨率）
    - num_heads：每个编码尺度的 transformer head 数（用于图像-事件跨模态注意力块）
    """
    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super(EFNet, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads

        # stage1 / stage2 的编码器（两套参数不共享）
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()

        # 两阶段的输入 stem：将 RGB 映射到 wf 通道
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        # 事件分支：event -> wf 通道
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            # 最后一层不下采样（保持最小分辨率）
            downsample = True if (i+1) < depth else False

            # stage1 编码块：带图像-事件 transformer 融合（num_heads[i]）
            self.down_path_1.append(
                UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, num_heads=self.num_heads[i])
            )
            # stage2 编码块：可启用 EMGC（仅在需要下采样的层使用）
            self.down_path_2.append(
                UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_emgc=downsample)
            )

            # ev encoder：产生与每个尺度对齐的 event 特征 ev[i]
            if i < self.depth:
                self.down_path_ev.append(
                    UNetEVConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope)
                )

            prev_channels = (2**i) * wf

        # 解码器：从最深层逐级上采样回到高分辨率
        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()

        # 对 skip feature 先做 3x3 conv（对齐/平滑），再与上采样特征拼接
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            prev_channels = (2**i) * wf

        # SAM：从 stage1 解码末端特征得到 out_1，并输出给 stage2 的引导特征 sam_feature
        self.sam12 = SAM(prev_channels)

        # stage2 输入拼接：stem(x2) 与 sam_feature 在通道维 concat，再 1x1 压回 prev_channels
        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
        # 最后重建：输出 residual（再与 image 相加得到最终 out_2）
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, event, mask=None):
        """
        输入：
        - x:     [B, 3, H, W] 图像（模糊图）
        - event: [B, ev_chn, H, W] 事件体素/堆叠事件帧
        - mask:  可选，[B, 1, H, W] 或可广播到特征通道的掩码（用于 EMGC）
        输出：
        - [out_1, out_2]：stage1 中间结果、stage2 最终结果
        """
        image = x  # 保留原图用于残差连接

        # ------------------------------------------------------------
        # 1) 事件编码器：产生多尺度 event 特征列表 ev
        #    ev[i] 的分辨率与 stage1 对应尺度的特征对齐（取决于 fuse_before_downsample）
        # ------------------------------------------------------------
        ev = []
        e1 = self.conv_ev1(event)  # [B, wf, H, W]
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth - 1:
                # e1_down：下采样后的特征；e1_up：下采样前（高分辨率）特征
                e1, e1_up = down(e1, self.fuse_before_downsample)
                # 若选择“下采样前融合”，则保存高分辨率 e1_up；否则保存下采样后 e1
                ev.append(e1_up if self.fuse_before_downsample else e1)
            else:
                # 最后一层不再返回 (down, up) 二元组
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)

        # ------------------------------------------------------------
        # 2) stage1：图像编码（融合 event）-> 解码 -> SAM 得到 out_1 与 sam_feature
        # ------------------------------------------------------------
        x1 = self.conv_01(image)  # [B, wf, H, W]
        encs, decs, masks = [], [], []

        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                # 编码阶段融合 event：EventImage_ChannelAttentionTransformerBlock(out, ev[i])
                x1, x1_up = down(
                    x1,
                    event_filter=ev[i],
                    merge_before_downsample=self.fuse_before_downsample
                )
                # encs 存储 skip（通常为下采样前的高分辨率特征）
                encs.append(x1_up)

                # mask 需要与各尺度特征对齐：每下采样一层，mask 也相应缩放
                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor=0.5**i))
            else:
                # 最深层：不再下采样，仅做特征提取/融合
                x1 = down(
                    x1,
                    event_filter=ev[i],
                    merge_before_downsample=self.fuse_before_downsample
                )

        # 解码：逐级上采样，并与对应的 skip 特征拼接
        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)  # decs 用于 stage2 的 EMGC 引导（对称层）

        # SAM：输出 stage1 重建 out_1，并生成给 stage2 使用的 sam_feature
        sam_feature, out_1 = self.sam12(x1, image)

        # ------------------------------------------------------------
        # 3) stage2：在 stage1 的 enc/dec 特征与可选 mask 的引导下细化重建
        # ------------------------------------------------------------
        x2 = self.conv_02(image)
        # 将 stage1 的 sam_feature 作为额外先验拼接到 stage2 输入
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))

        blocks = []  # 保存 stage2 编码产生的 skip（供 stage2 解码使用）
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                # EMGC 仅在提供 enc/dec/mask 时启用（见 UNetConvBlock 内逻辑）
                if mask is not None:
                    x2, x2_up = down(x2, encs[i], decs[-i-1], mask=masks[i])
                else:
                    x2, x2_up = down(x2, encs[i], decs[-i-1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]))

        # 输出 residual 并与输入图像相加得到最终结果
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


class UNetConvBlock(nn.Module):
    """
    U-Net 编码/解码中的基础卷积块（带残差），可选能力：
    1) downsample=True：块末尾用 stride=2 卷积下采样，并返回 (out_down, out_skip)
    2) use_emgc=True：当提供 enc/dec/mask 时启用 EMGC，对 out 做额外的编码/解码引导修正
    3) num_heads!=None：构建图像-事件跨模态注意力模块（EventImage_ChannelAttentionTransformerBlock）

    forward 输入（按需提供）：
    - x: 主干特征 [B, C_in, H, W]
    - enc/dec: 来自 stage1 的对称层特征（用于 EMGC）
    - mask: 对应尺度的掩码（用于分区引导：mask=1 区域偏向 dec，mask=0 区域偏向 enc）
    - event_filter: 对应尺度的 event 特征（用于跨模态融合）
    - merge_before_downsample: 决定融合发生在下采样前还是下采样后
    """
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None):
        super(UNetConvBlock, self).__init__()
        # 注意：这里 self.downsample 初始为 bool，若 downsample=True 会在后面被替换成 conv_down 模块
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)  # 用于残差支路的通道对齐
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        # 两层 3x3 卷积 + LeakyReLU（基础特征提取）
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        # EMGC：编码引导/解码引导各一套卷积，同时对 mask/1-mask 做加权分支
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            # 用 4x4 stride=2 卷积替代 bool 标记（下采样）
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.num_heads is not None:
            # 图像-事件融合：用 transformer block 将 event 信息注入到图像特征 out
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(
                out_size,
                num_heads=self.num_heads,
                ffn_expansion_factor=4,
                bias=False,
                LayerNorm_type='WithBias'
            )

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        # 基础卷积块：Conv -> 激活 -> Conv -> 激活，并加残差
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)

        # --------------------------
        # EMGC：利用 stage1 的 enc/dec + mask 对 out 做区域化补偿
        # - mask==1 区域更相信 dec（解码语义/重建）
        # - mask==0 区域更相信 enc（编码纹理/细节）
        # 仅当 enc/dec/mask 都给出时启用
        # --------------------------
        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1 - mask) * enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask * dec)
            out = out + out_enc + out_dec

        # --------------------------
        # 图像-事件融合（跨模态注意力）
        # - merge_before_downsample=True：在高分辨率 out 上融合
        # - 否则在下采样后的 out_down 上融合
        # --------------------------
        if event_filter is not None and merge_before_downsample: # 在下采样前融合
            out = self.image_event_transformer(out, event_filter)

        if self.downsample:
            out_down = self.downsample(out) # 融合后做下采样
            if not merge_before_downsample: # 有event，但是在下采样后融合
                out_down = self.image_event_transformer(out_down, event_filter)
            # 返回：下采样后的特征 + 供 skip 使用的高分辨率特征
            return out_down, out
        else: 
            if merge_before_downsample: # 有event，并且在下采样前融合，实际不存在？
                return out
            else:
                # 这里是“最后一层不下采样但希望在该分辨率融合”的情况
                out = self.image_event_transformer(out, event_filter)
                return out  # 补全返回，避免隐式返回 None

class UNetEVConvBlock(nn.Module):
    """
    事件分支的卷积块：结构类似 UNetConvBlock，但不做图像-事件融合，仅生成多尺度 event 特征。
    额外提供 conv_before_merge（1x1）用于在“融合前/融合后”两种策略下对齐 event 特征分布：
    - merge_before_downsample=True：对下采样前 out 做 1x1
    - merge_before_downsample=False：对下采样后 out_down 做 1x1
    """
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0) # kernel stride padding
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
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
            
            # 根据融合发生位置，决定对哪一路做 1x1 对齐
            if not merge_before_downsample:
                # 融合在下采样后发生：对 out_down 做变换，供后续与图像特征对齐融合
                out_down = self.conv_before_merge(out_down)
            else:
                # 融合在下采样前发生：对 out（高分辨率）做变换，作为该尺度的 event_filter
                out = self.conv_before_merge(out)
            return out_down, out
        else:
            # 最深层：不再下采样，直接输出变换后的 event 特征
            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):
    """
    U-Net 上采样块：
    1) 使用转置卷积将特征上采样到高一尺度（H,W *2，通道变为 out_size）
    2) 与对应尺度的 skip（bridge）在通道维拼接
    3) 通过 UNetConvBlock（downsample=False）融合拼接后的特征
    """
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope) # relu_slope：leak_relu 负数的斜率

    def forward(self, x, bridge):
        # x: [B, C_in, H, W] -> up: [B, C_out, 2H, 2W]
        up = self.up(x)
        # bridge: 同尺度 skip 特征 [B, C_out, 2H, 2W]
        out = torch.cat([up, bridge], 1)  # [B, 2*C_out, 2H, 2W]
        out = self.conv_block(out)
        return out


if __name__ == "__main__":
    pass
