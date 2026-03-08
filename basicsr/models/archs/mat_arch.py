"""
MAT_arch.py

该文件实现 MAT 模型（图像-事件融合的 Transformer/U-Net 混合架构），主要包含：
1) 基础组件：LayerNorm、FFN、Attention 等 Transformer 常用模块；
2) 事件分支：UNetEVTransformerBlock + AMMP 预测稀疏mask，用于事件特征的选择/增强；
3) 主干网络：三层编码器 + 三层解码器 + refinement，将事件特征与图像特征在编码阶段融合；
4) 工具函数：非零占比统计 get_non_zero_ratio、topk2D 等。

输入约定（MAT.forward）：
- x: (B, 9, H, W)
  - x[:, 0:3] 为 RGB 图像
  - x[:, 3:9] 为 6 通道事件体素/事件帧（bins=6）
输出：
- (B, 3, H, W) 重建/增强后的图像（残差形式：pred + inp_img）

注：本文件包含若干调试/统计相关 import（如 cv2/plt/torchvision），训练部署时如需精简可再行清理。
"""

import math
from torch.autograd import Variable
import einops
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import numpy as np
from einops import rearrange
import torchvision
import cv2


def get_non_zero_ratio(x: torch.Tensor) -> torch.Tensor:
    """
    统计事件张量在不同空间尺度上的“非零占比”（稀疏程度），用于后续动态控制/门控。

    参数:
        x: (B, C, H, W)，通常 C=6 (bins)
    返回:
        ratios: list[Tensor]，长度为3，对应 1x / 2x / 4x 下采样尺度
            - 每个元素形状: (B, C)
            - 值域: [0, 1] 左右（按 numel 比例归一化），数值越大表示该 bin 越“密集”

    实现说明:
        - 先对空间维度做 max_pool 下采样，模拟更大感受野统计
        - 使用 (x!=0) 计数并按总元素数归一化

    备注（与本网络 depth 的对应关系）：
        - 本实现返回 3 个尺度（原尺度/2x/4x），恰好对应 MAT 中 depth=3 的事件分支层数：
          ratio[0] -> 第1层（H,W）
          ratio[1] -> 第2层（H/2,W/2）
          ratio[2] -> 第3层（H/4,W/4）
    """
    # Downsample to match the receptive field of each SAST block.
    x_down_2 = torch.nn.functional.max_pool2d(x.float(), kernel_size=2, stride=2)
    x_down_4 = torch.nn.functional.max_pool2d(x_down_2, kernel_size=2, stride=2)
    # Count the number of non-zero elements in each bin.
    num_nonzero_1 = torch.sum(torch.sum(x != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_2 = torch.sum(torch.sum(x_down_2 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)
    num_nonzero_3 = torch.sum(torch.sum(x_down_4 != 0, dtype=torch.int16, dim=[2]), dtype=torch.int16, dim=-1)

    result1 = x.shape[0] / x.numel() * num_nonzero_1.float()
    result2 = x.shape[0] / x_down_2.numel() * num_nonzero_2.float()
    result3 = x.shape[0] / x_down_4.numel() * num_nonzero_3.float()
    # Return the ratio of non-zero elements in each bin at four scales.
    return [abs(result1), abs(result2), abs(result3)]


class PositiveLinear(nn.Module):
    """
    权重强制为正的线性层：W = exp(raw_W)

    使用场景：
        - 用于将稀疏比例 ratio 映射为控制向量（scale/gate），避免出现负权重导致门控方向不稳定。

    输入:
        input: (..., in_features)
    输出:
        (..., out_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # 使用 exp 保证权重为正；注意可能带来数值放大，训练时可结合初始化/学习率稳定
        positive_weights = torch.exp(self.weight)
        return nn.functional.linear(input, positive_weights, self.bias)


class LayerNormProxy(nn.Module):
    """
    针对 (B,C,H,W) 的 LayerNorm 代理封装：
    - 内部将张量转换到 channel-last (B,H,W,C) 做 nn.LayerNorm(C)
    - 再转回 channel-first

    适用：
        - 想使用 PyTorch 原生 LayerNorm 但输入是 NCHW 的场景。
    """
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')



##########################################################################
## Layer Norm
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    """便捷封装：带 same-padding 的 Conv2d（padding=kernel//2）。"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def ChannelSplit(inp):
    """
    沿 channel 维度均分为两半。
    常用于：把 [image_feat, event_feat] 拼接后的特征拆回图像/事件部分。
    """
    C = inp.shape[1]
    return inp[:, :C//2, :, :], inp[:, C//2:, :, :]

def to_3d(x):
    """将 (B,C,H,W) 展平为 (B, H*W, C)，便于做 token 级 LN/attention。"""
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    """将 (B, H*W, C) 还原为 (B,C,H,W)。"""
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """
    无偏置 LayerNorm（仅缩放不平移）：
    - 只估计方差 sigma，不减均值 mu
    - 在某些视觉任务中可更稳定/更省参数
    """
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """
    带偏置 LayerNorm（标准 LN）：
    - 估计 mu 和 sigma，并做仿射变换
    """
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
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
    """
    NCHW 版本 LayerNorm 统一入口：
    - 先 to_3d 变为 token 序列
    - 做 BiasFree/WithBias LN
    - 再 to_4d 还原
    """
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    """
    Transformer FFN（卷积版）：
    - 1x1 升维到 2*hidden
    - 3x3 depthwise conv 引入局部上下文
    - chunk 分成两支做 gated gelu：GELU(x1) * x2
    - 1x1 降回 dim

    输入/输出:
        x: (B, dim, H, W) -> (B, dim, H, W)
    """
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


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    """
    标准多头自注意力（MDTA 变体，使用卷积产生 qkv，并以 (H*W) 作为 token 维）：

    关键点：
    - qkv: 1x1 conv 生成后再接 depthwise 3x3 conv，增强局部建模
    - q,k 做 L2 normalize（在视觉注意力中常用于稳定点积范围）
    - temperature 为每个 head 可学习缩放

    输入/输出:
        x: (B, C, H, W) -> (B, C, H, W)
    """
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

        # q1 = self.CS(q)
        # k1 = self.CS(k)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # visualize_attention_matrix(attn.squeeze(0).squeeze(0).data.cpu())
        attn = attn.softmax(dim=-1)


        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out
    
class ChannelPool(nn.Module):
    """通道池化：拼接 (max_pool, mean_pool) -> (B,2,H,W)，常用于通道注意力/压缩。"""
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class MSA(nn.Module):
    """
    Masked Self-Attention（用于事件编码器 evs_enc=True 分支）：

    与 Attention 的区别：
    - 在计算注意力前对 q,k 乘 mask（mask 通常为 (B,1,1,H*W) 广播到 head/channel）
    - 用于只在“选中的空间位置”上建立强关联

    forward(x, mask):
        x: (B,C,H,W)
        mask: (B,1,1,H*W) 或可广播到 (B,head,c,H*W)
    """
    def __init__(self, dim, num_heads, bias):
        super(MSA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.deconv = DeformConv2D(dim, dim*3, 3, 1)

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
        
        # self.para_proj = nn.Linear(, )


    def forward(self, x, mask):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        q = q*mask
        k = k*mask
       
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)

        return out
    


class MAA(nn.Module):
    """
    Modulated Attention with Events（图像主干 use_evs=True 时使用）：

    作用：
    - 使用外部调制量 ME（来自事件分支/门控）对 q 做增强/抑制
    - 直观理解：事件告诉注意力“哪里更该关注”

    forward(x, ME):
        x: (B,C,H,W)
        ME: 形状需可广播到 q 的形状 (B,head,c,H*W)
            在当前工程中由 se/mask 类信号传入（注意 shape 对齐）
    """
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
       
    # origin
    def forward(self, x, ME):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        in_ME = -1 * (ME - 1)
        q_enhe = ME * self.temperature_e + in_ME
        q = q*q_enhe

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        del q,k
        attn = attn.softmax(dim=-1)

        out = attn @ v
        del v,attn
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out

   
class CMIG(nn.Module):
    """
    Cross-Modal Interaction Gate（跨模态门控融合）：

    作用：
    - 将事件特征 ev 作为门控信号，调制图像特征 x（element-wise）
    - 形式：x <- GELU(ev) * x，然后 1x1 conv 混合通道

    forward(x, ev):
        x, ev: (B, C, H, W) 且 C 对齐（上层通过拼接后切分保证维度一致）
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(CMIG, self).__init__()

        # g0
        self.dim = dim
        hidden_features = dim
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def sort_feature(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        avg_values = avg_pool.view(x.size(0), x.size(1))
        sorted_indices = avg_values.argsort(dim=1)
        # print(sorted_indices)
        x[:,:,:,:] = x[:, sorted_indices,:,:]
        return x

    def forward(self, x, ev):
        # g0
        x = F.gelu(ev) * x
        x = self.project_out(x)
        return x


class TransformerBlock_MAT(nn.Module):
    """
    MAT 的基础 Transformer Block，支持三种工作模式：
    1) use_evs=False, evs_enc=False: 普通 Attention + FFN（用于解码器/图像无事件块）
    2) use_evs=True: 使用 CMIG 融合事件 -> 使用 MAA（事件调制注意力）-> FFN（用于图像编码器）
    3) evs_enc=True: 使用 MSA（带 mask 的注意力）-> FFN（用于事件分支编码）

    forward(x, Mask_E=None):
        - 当 use_evs=True 时，输入 x 实际为 concat([img_feat, ev_feat], dim=1)
          并在内部切分后融合；返回时再拼回，便于 CustomSequential 透传事件缓存。
        - Mask_E: 事件分支产生的稀疏mask/调制量，用于 MSA 或 MAA。
    """
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

    def forward(self, x, Mask_E=None):
        # use_evs=True 时：输入通道一分为二，前半为图像特征，后半为事件特征（在本 block 内部短路保存 events）
        if self.use_evs:
            C = x.shape[1]
            # print(x.shape)
            events = x[:, C//2:, :, :]
            x = x[:, :C//2, :, :]
            
            x = self.fusion(self.norm0(x), self.norm_evs(events)) + x
        if Mask_E is None:
            x = x + self.attn(self.norm1(x))
        else:
            x = x + self.attn(self.norm1(x), Mask_E)
        # save_as_heatmap(x[:, 0, :, :], f"xvis_{x.size(2)}_{1}")
        
        x = x + self.ffn(self.norm2(x))
        if self.use_evs:
            return torch.cat([x, events], 1), Mask_E  
        else: 
            return x
        
            

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    """
    重叠式 patch embedding：
    - 使用 3x3 conv stride=1 生成初始特征
    - 与 ViT 的“非重叠 patchify”不同，这里保留像素级分辨率，更适合低层视觉任务。
    """
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x

def conv_down(in_chn, out_chn, bias=False):
    """下采样卷积：kernel=4, stride=2 的 Conv2d（常见于 U-Net）。"""
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    """
    主干下采样模块（Restormer 风格）：
    - 先 3x3 conv 将通道减半
    - PixelUnshuffle(2) 做 2x 下采样（空间 /2，通道 x4），总体通道变为 n_feat*2
    """
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """
    主干上采样模块：
    - 先 3x3 conv 将通道扩到 2x
    - PixelShuffle(2) 做 2x 上采样（空间 x2，通道 /4）
    """
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
    

def topk2D(
    input: torch.Tensor,
    k: torch.Tensor,
    largest: torch.Tensor
) -> torch.Tensor:
    """
    批量 topk 索引选择的辅助函数（支持每个 batch 样本不同 k）：
    - 先用 max_k 做一次 topk 得到候选索引
    - 再用取模/聚合的方式为每个样本选择其对应 k 的索引子集

    注意：
        当前文件中训练分支的 topk2D 用法已被注释，保留函数主要用于实验/回滚。
    """
    max_k = k.max().int()
    # print(k)
    # input = input.squeeze(1).squeeze(1)
    fake_indexes = torch.topk(input, max_k, dim=-1, largest=largest).indices
    # torch.cuda.synchronize()
    T = torch.arange(max_k).expand_as(fake_indexes).to(device=input.device)
    T = torch.remainder(T, k.unsqueeze(1))
    
    indexes = torch.gather(fake_indexes, -1, T)
    # print(indexes.shape)
    return indexes

# import faiss
class AMMP(nn.Module):
    """
    AMMP：事件分支的“得分/掩码预测器”（用于生成稀疏mask + 特征加权系数 weighting）

    核心输入:
        x: (B, C, H, W) 事件特征
        mask: (B,1,1,H*W) 上一层的 decision/mask（初始全 1）
        ratio: (B, bins) 事件稀疏比例（来自 get_non_zero_ratio），用于动态调节门控强度

    输出:
        new_mask: (B,1,1,H*W)，由得分 output 经过 topk 双向选择（最大/最小）得到的二值mask
        weighting: (B, C, H, W) 或可广播的权重图（此处实现为 sigmoid(alpha)*GELU(...)）

    实现细节:
        - local_x: 1x1 conv 提取局部特征
        - global_x: 在 mask 选择区域上做全局聚合（加权平均）
        - scale: ratio -> PositiveLinear -> 通道/分支控制向量
        - alpha: 用 ratio.max() 做上界更新（类似论文中的 beta），并用于计算 K（topk 比例）
        - topk 同时选 largest 与 smallest：相当于保留“强响应”和“弱响应”两类位置（实现上是并集）
    """
    def __init__(self, embed_dim, LayerNorm_type):
        super().__init__()

        self.in_norm = LayerNorm(embed_dim, LayerNorm_type)
        self.bins = 6
        self.to_controls = PositiveLinear(self.bins, embed_dim*2, bias=False)

        # conv head (chd)
        self.in_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.GELU()
        )

        self.conv_compress = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim//2, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim//2, 2, bias=False),
            nn.GELU()
        )
        self.channel_compress = nn.Sequential(
            ChannelPool(),
            nn.Conv2d(2, 2, kernel_size=1, bias=False),
            nn.GELU()
        )

        self.in_compress3 = nn.Sequential(
            nn.Linear(4, 1, bias=False),
            nn.GELU()
        )
        # self.out_proj = nn.LogSoftmax(dim=-1)
        
        self.alpha = nn.Parameter(torch.ones(1, 1, 1)*0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask, ratio):
        B, C, H, W = x.size()
        x_= x
        x = self.in_norm(x)
        # conv head (chd)
        local_x = self.in_proj(x)

        ratio = ratio[:, None, None, :] # [B,1,1,Bin]
        scale = self.to_controls(ratio)

        # Global Agg
        mask = rearrange(mask, "b head c (h w) -> b (head c) h w", head=1, h=H, w=W)
        global_x = (local_x * mask).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True) / (torch.sum(torch.sum(mask, dim=-1, keepdim=True), dim=-2, keepdim=True))
        global_x[global_x==torch.inf] = 0
        # Channel Compress
        x_1 = self.channel_compress(local_x*global_x)
        x_1 = rearrange(x_1, "b (i c) h w -> b i (h w) c", c=2, h=H, w=W)

        x_2 = torch.cat([local_x, global_x.expand(B, C, H, W)], dim=1) * scale.permute(0, -1, 1, 2)
        x_2 = rearrange(x_2, "b (i c) h w -> b i (h w) c", c=2*C, h=H, w=W)
        x_2 = self.conv_compress(x_2)
        
        x = self.in_compress3(torch.cat([x_1, x_2], dim=-1))
        output = rearrange(x, "b (head c) h w -> b head c (h w)", head=1) 

        # -------------------topk2d_for training-------------------
        # mu_r = torch.mean(ratio, dim=-1) 
        # b = torch.ones(mu_r.shape, device=output.device)*0.4
        # m = mu_r/self.alpha
        # K = torch.where(m >= b, b, m)
        # K = torch.where(K <= mu_r, mu_r, K)
        # K = torch.where(K <= b*0.01, 0.005, K)
        # K = K*H*W
        # K = K.int()
        # print(self.alpha)
        # indexs = topk2D(output, k=K, largest=True)
        # new_mask = torch.zeros(B, 1, 1, H*W, device=x.device, requires_grad=False)
        # new_mask.scatter_(-1, indexs, 1.)
        # indexs = topk2D(output, k=K, largest=False)
        # new_mask.scatter_(-1, indexs, 1.)
        # -------------------topk2d_for traning-------------------

        # ratio_scale (rs)   alpha == beta in paper
        self.alpha = ratio.max() if self.alpha < ratio.max() else self.alpha
        m = ratio.max()/self.alpha
        K = m if m<=0.4 else 0.4
        K = torch.where(K <= 0.004, 0.005, K)

        # topk_maxr for testing
        indexs = torch.topk(output, k=int(K*H*W), dim=-1, largest=True, sorted=False)[1]
        new_mask = torch.zeros(B, 1, 1, H*W, device=x.device, requires_grad=False)
        new_mask.scatter_(-1, indexs, 1.)
        indexs = torch.topk(output, k=int(K*H*W), dim=-1, largest=False, sorted=False)[1]
        new_mask.scatter_(-1, indexs, 1.)
        
        # (rs)6
        return new_mask, self.sigmoid(self.alpha)*F.gelu(rearrange(x, "b i (h w) c -> b (i c) h w", c=1, h=H, w=W))
    

class UNetEVTransformerBlock(nn.Module):
    """
    事件分支的 U-Net 风格块（编码侧）：
    - conv_1: 调整通道到 out_size
    - score_predictor(AMMP): 预测稀疏 mask + weighting，对特征做增强（ECSG 逻辑）
    - encoder: 若干 TransformerBlock_MAT(evs_enc=True) 用 mask 做 MSA
    - identity: 残差支路，保证信息直通
    - 可选 downsample: 进入下一层事件尺度

    forward 返回:
        - 若 downsample=True:
            out_down: 下采样后的特征（供下一层）
            out:      下采样前/融合后的特征（供与图像分支融合或 skip）
            mask:     当前层的 decision mask
        - 若 downsample=False:
            out, mask
    """
    def __init__(self, in_size, out_size, downsample, num_heads):
        super(UNetEVTransformerBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=1, padding=0, bias=True)
    
        self.score_predictor = AMMP(out_size, LayerNorm_type="WithBias")
        self.encoder = [TransformerBlock_MAT(evs_enc=True, dim=out_size, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias") for _ in range(2)]
        self.encoder = nn.Sequential(*self.encoder)

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, ratio=None, merge_before_downsample=True):

        B, C, H, W = x.size()
        prev_decision = torch.ones(B, 1, 1, H*W, dtype=x.dtype, device=x.device)  
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
            else : 
                out = self.conv_before_merge(out)
            return out_down, out, mask
        else:
            out = self.conv_before_merge(out)
            return out, mask


class CustomSequential(nn.Sequential):
    """
    自定义 Sequential：用于传递 (feat, mask) 的二元组。
    - 每个 module 的 forward 签名需形如 module(x, mask) -> (x, mask) 或 x
    - 这里配合 TransformerBlock_MAT(use_evs=True) 的返回形式使用
    """
    def forward(self, x):
        for module in self._modules.values():
            x1, x2 = x
            x = module(x1, x2)
        return x
    
class MAT(nn.Module):
    """
    MAT 主网络：
    - 图像分支：三层 encoder（每层多个 TransformerBlock_MAT，且 use_evs=True 进行事件融合）
    - 事件分支：UNetEVTransformerBlock 逐层下采样，生成 ev 特征与 se(mask/调制信号)
    - 解码器：逐层上采样并与 encoder skip 连接融合，最后 refinement 并输出 residual

    关键数据流（简化）：
        events -> conv_ev1 -> down_path_ev -> 得到 ev[i] & se[i]
        img -> patch_embed -> concat(ev[0]) -> encoder_level1(se[0]) -> ...
        最终 output + inp_img

    注意：
        - encoder_level* 使用 CustomSequential 来在 block 间透传 se(mask)
        - ChannelSplit 用于把拼接的 [img_feat, ev_feat] 拆回图像特征（只取前半进入后续）
    """
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[8, 8, 7],
        num_refinement_blocks=2,
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):

        super(MAT, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = CustomSequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**0),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    use_evs=True
                )
                for i in range(num_blocks[0])
            ]
        ) 
  

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = CustomSequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    use_evs=True
                )
                for i in range(num_blocks[1])
            ]
        ) 
      

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
  
        self.encoder_level3 = CustomSequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    use_evs=True
                )
                for i in range(num_blocks[2])
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
                for i in range(num_blocks[2])
            ]
        )


        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(
            int(dim * 2**1), int(dim), kernel_size=1, bias=bias
        )
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock_MAT(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
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
                for i in range(num_refinement_blocks)
            ]
        )
        self.down_path_ev = nn.ModuleList()
        prev_channels = dim
        depth = len(num_blocks)
        self.depth = depth
        for i in range(depth):
            downsample = True if (i+1) < depth else False 
            # ev encoder
            if i < self.depth:
                self.down_path_ev.append(UNetEVTransformerBlock(prev_channels, (2**i) * dim, downsample,num_heads=heads[i]))

            prev_channels = (2**i) * dim

        self.output = nn.Conv2d(
            int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.conv_ev1 = nn.Conv2d(6, dim, 3, 1, 1)
        self.fuse_before_downsample = True

        
    def forward(self, x, event):
        """
        x: (B, 9, H, W)
        返回:
            out: (B, 3, H, W)

        关键中间量（便于调试/二次开发）：
            - ev[i]：事件分支在第 i 层输出、用于与图像特征 concat 的事件特征 (B, dim*2^i, H/2^i, W/2^i)
            - se[i]：事件分支产生的稀疏 decision mask（用于编码器块的 MAA/MSA 调制）
                    形状通常可广播到注意力的 token 维（实现里按 (B,1,1,H*W) 传递）
        """
        inp_img = x
        events = event
        ratio = get_non_zero_ratio(events) # [B, bin]

        ev = []
        se = []
        e1 = self.conv_ev1(events)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth-1:
                e1, e1_up, score = down(e1, ratio[i], self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1, score = down(e1, ratio[i], self.fuse_before_downsample)
                ev.append(e1)
            se.append(score)


        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1 = torch.cat([inp_enc_level1, ev[0]], 1)
        # print(inp_enc_level1.shape)
        out_enc_level1, _ = self.encoder_level1((inp_enc_level1, se[0]))
        out_enc_level1 = ChannelSplit(out_enc_level1)[0]
        del inp_enc_level1
        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_level2 = torch.cat([inp_enc_level2, ev[1]], 1)
        out_enc_level2, _ = self.encoder_level2((inp_enc_level2, se[1]))
        out_enc_level2 = ChannelSplit(out_enc_level2)[0]
        del inp_enc_level2
        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc_level3 = torch.cat([inp_enc_level3, ev[2]], 1)
        
        out_enc_level3, _ = self.encoder_level3((inp_enc_level3, se[2]))
        inp_dec_level3 = ChannelSplit(out_enc_level3)[0]

        del ev, out_enc_level3, inp_enc_level3
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        del out_enc_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        del out_enc_level1
        out_dec_level1 = self.refinement(out_dec_level1)


        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


from calflops import calculate_flops
# pip install calflops transformers
if __name__ == "__main__":
    """
    仅用于本地统计 FLOPs/MACs/Params 的入口：
    - 不影响作为模块被 import
    - 如用于训练/推理部署，可移除此段或放到独立脚本
    """
    model = MAT()
    # print(model)
    # input = torch.randn(1, 9, 128, 128)
    # output = model(input)
    # print("-" * 50)
    # print(output.shape)

    # from thop import profile
    # flops, params = profile(model, inputs=torch.randn(1, 1, 9, 1024, 728))
    # print(" FLOPs:%s    Params:%s \n" %(flops, params))


    input_shape = (1, 9, 1024, 728)
    flops, macs, params = calculate_flops(
        model=model, input_shape=input_shape, output_as_string=True, output_precision=4
    )
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
