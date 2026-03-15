import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

#-----------------layer norm-----------------------------
def to_3d(x):
    """将 4D 特征 (B,C,H,W) 展平为空间 token 序列 (B, H*W, C)，便于做 LayerNorm。"""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """将 token 序列 (B, H*W, C) 还原为 4D 特征 (B,C,H,W)。"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """无偏置 LayerNorm：仅使用方差归一化（不减均值），再乘可学习缩放 weight。"""
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
    """带偏置 LayerNorm：标准 LN（减均值/除方差），再做仿射变换（weight + bias）。"""
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
    """对 2D 特征图做 LayerNorm：内部先展平空间维度，再还原回 (B,C,H,W)。"""
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

#-------------------------layer norm end-----------------------------------

class DFFN(nn.Module):
    """
    DFFN：在局部 patch 上做频域调制的前馈网络（类似 FFN，但引入 rFFT/irFFT）。
    - 先 1x1 升维得到 2*hidden
    - 以 patch_size 切块后做 rfft2 -> 频域逐元素缩放(可学习参数 self.fft) -> irfft2
    - 深度可分离卷积后按通道一分为二，用 GELU 门控融合，再 1x1 投回 dim
    """
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: (B, dim, H, W)
        x = self.project_in(x)  # (B, 2*hidden, H, W)

        # 将特征按 patch_size 切成不重叠块，便于在每个小块上做 2D FFT
        x_patch = rearrange(
            x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=self.patch_size, patch2=self.patch_size
        )

        # rfft2：实数输入的 2D FFT（频域最后一维会变为 patch_size//2+1）
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        # 可学习的频域缩放/滤波（逐通道逐频点）
        x_patch_fft = x_patch_fft * self.fft
        # 逆变换回空间域 patch
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        # 拼回原图大小
        x = rearrange(
            x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
            patch1=self.patch_size, patch2=self.patch_size
        )

        # 深度卷积后做门控：GELU(x1) * x2
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2

        x = self.project_out(x)  # (B, dim, H, W)
        return x

class FSAS(nn.Module):
    """
    FSAS：频域自注意力风格模块（用频域相关替代显式 QK^T）。
    - 1x1 得到 6*dim，再 depthwise 3x3，切成 q,k,v (各 2*dim)
    - q/k 以 patch 切块后做 rfft2，相乘得到“频域相关”，再 irfft2 回空间
    - 对相关结果做 LayerNorm，再与 v 相乘并 1x1 投回 dim
    """
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        # x: (B, dim, H, W)
        hidden = self.to_hidden(x)  # (B, 6*dim, H, W)

        # 切 patch 后做频域变换
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)  # 各 (B, 2*dim, H, W)

        q_patch = rearrange(
            q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=self.patch_size, patch2=self.patch_size
        )
        k_patch = rearrange(
            k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=self.patch_size, patch2=self.patch_size
        )
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        # 频域逐元素相乘，相当于在每个 patch 上做相关/匹配（实现上更轻量）
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))

        # 拼回空间并做归一化
        out = rearrange(
            out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
            patch1=self.patch_size, patch2=self.patch_size
        )
        out = self.norm(out)

        # 用相关结果对 v 做调制，再映射回 dim
        output = v * out
        output = self.project_out(output)
        return output

#-------------------------event block-----------------------------
class EventFeedForward(nn.Module):
    """通用 FFN（支持 expansion=0 的轻量模式，以及门控式 FFN）。"""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EventFeedForward, self).__init__()
        self.ffn_expansion_factor = ffn_expansion_factor
        self.dim = dim
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                    groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, evt):
        x = self.project_in(evt)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        return x
 
class EventFSAS(nn.Module):
    def __init__(self, dim, bias):
        super(EventFSAS, self).__init__()

        self.q = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        
        self.q_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        
        self.norm1_image = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm1_event = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x, evt):
        # x: (B, dim, H, W)
        assert x.shape == event.shape, "x and event must have the same shape"
        x = self.norm1_image(x)
        evt = self.norm1_event(evt)
        
        q = self.q_dwconv(self.q(x))  # (B, 2*dim, H, W)
        kv = self.kv_dwconv(self.kv(evt))  # (B, 4*dim, H, W)
        k, v = kv.chunk(2, dim=1)  # 各 (B, 2*dim, H, W)

        q_patch = rearrange(
            q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=self.patch_size, patch2=self.patch_size
        )
        k_patch = rearrange(
            k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=self.patch_size, patch2=self.patch_size
        )
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        # 频域逐元素相乘，相当于在每个 patch 上做相关/匹配（实现上更轻量）
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))

        # 拼回空间并做归一化
        out = rearrange(
            out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
            patch1=self.patch_size, patch2=self.patch_size
        )
        out = self.norm(out)

        # 用相关结果对 v 做调制，再映射回 dim
        output = v * out
        output = self.project_out(output)
        return output
    
class Event_Transformer_Block(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias', att=False):
        super(Event_Transformer_Block, self).__init__()
        self.att = att
        if self.att:
            raise NotImplementedError("Event_Transformer_Block does not support att=True currently.")

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = EventFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, evt):
        evt = evt + self.ffn(self.norm2(evt))
        return evt
    
class Event_Image_Fuse(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Event_Image_Fuse, self).__init__()

        self.attn = EventFSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, evt):
        # 残差连接：x <- x + Attn(LN(x))
        x = x + self.attn(x, evt)

        # 残差连接：x <- x + FFN(LN(x))
        x = x + self.ffn(self.norm2(x))
        return x
#------------------------event Block end-----------------------------------
class TransformerBlock(nn.Module):
    """
    TransformerBlock：残差结构
    - 可选注意力分支（att=True 时启用 FSAS）
    - 必选 DFFN 分支
    """
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias', att=False):
        super(TransformerBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # 残差连接：x <- x + Attn(LN(x))
        if self.att:
            x = x + self.attn(self.norm1(x))

        # 残差连接：x <- x + FFN(LN(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Fuse(nn.Module):
    """编码器/解码器特征融合：concat -> 1x1 -> TransformerBlock -> 1x1 -> 分半相加。"""
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerBlock(dim=n_feat * 2)

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        # enc/dnc: (B, n_feat, H, W)，先通道拼接为 (B, 2*n_feat, H, W)
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.att_channel(x)
        x = self.conv2(x)
        # 拆分回两支并相加，得到融合结果 (B, n_feat, H, W)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        output = e + d
        return output


class OverlapPatchEmbed(nn.Module):
    """重叠 patch 嵌入：用 3x3 Conv 将输入映射到 embed_dim（保持分辨率不变）。"""
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    """下采样：先双线性缩小到 1/2，再用 Conv 将通道翻倍（n_feat -> 2*n_feat）。"""
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """上采样：先双线性放大到 2 倍，再用 Conv 将通道减半（n_feat -> n_feat/2）。"""
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)

    
# fft_cross_modal_attention   
class eventfft_cross(nn.Module):
    def __init__(self,
                 evt_inp=6,
                 img_inp=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 4, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(eventfft_cross, self).__init__()

        self.evt_embed = OverlapPatchEmbed(evt_inp, dim)
        self.img_embed = OverlapPatchEmbed(img_inp, dim)

        # event encoder
        self.evt_encoder_level1 = nn.Sequential(*[
            Event_Transformer_Block(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in
            range(num_blocks[0])])
        self.evt_down1_2 = Downsample(dim)
        
        self.evt_encoder_level2 = nn.Sequential(*[
            Event_Transformer_Block(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])
        self.evt_down2_3 = Downsample(int(dim * 2 ** 1))
        
        self.evt_encoder_level3 = nn.Sequential(*[
            Event_Transformer_Block(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])
        # image encoder
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])
        # image event cross attention fuse
        self.fuse_en1 = Event_Image_Fuse(dim, ffn_expansion_factor, bias)
        self.fuse_en2 = Event_Image_Fuse(dim * 2, ffn_expansion_factor, bias)
        self.fuse_en3 = Event_Image_Fuse(dim * 4, ffn_expansion_factor, bias)
        # decoder
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_refinement_blocks)])
        self.fuse_en1 = Fuse(dim)
        self.fuse_en2 = Fuse(dim * 2)
        self.fuse_en3 = Fuse(dim * 4)
        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, event):
        # 输入
        evt_inp = self.evt_embed(event)
        img_inp = self.img_embed(x)
        # event encoder
        evt_out_enc_level1 = self.evt_encoder_level1(evt_inp)
        
        evt_inp_enc_level2 = self.evt_down1_2(evt_out_enc_level1)
        evt_out_enc_level2 = self.evt_encoder_level2(evt_inp_enc_level2)
        
        evt_inp_enc_level3 = self.evt_down2_3(evt_out_enc_level2)
        evt_out_enc_level3 = self.evt_encoder_level3(evt_inp_enc_level3)
        # image encoder and fuse
        out_enc_level1 = self.encoder_level1(img_inp)
        out_enc_level1 = self.fuse_en1(out_enc_level1, evt_out_enc_level1) # 下采样前

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2 = self.fuse_en2(out_enc_level2, evt_out_enc_level2) # 下采样前

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc_level3 = self.fuse_en3(out_enc_level3, evt_out_enc_level3) # 下采样前

        # 解码最高层（带注意力）
        out_dec_level3 = self.decoder_level3(out_enc_level3)

        # 上采样并与 encoder 的同尺度特征融合（skip connection）
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # 细化阶段
        out_dec_level1 = self.refinement(out_dec_level1)

        # 输出层 + 全局残差（有助于学习残差图，稳定训练）
        out_dec_level1 = self.output(out_dec_level1) + x
        return out_dec_level1
    
