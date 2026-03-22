"""EventFFTViT5 — Dual-stream event-guided image restoration with ViT-5 modules.

Full ViT-5 refactor: replaces all FFT-based modules with standard transformer
components from ViT-5 (reference_2, arXiv 2026-02):
  - WindowSelfAttention: windowed dot-product attention + QK Norm + 2D RoPE
  - WindowCrossAttention: windowed cross-modal attention + QK Norm
  - Mlp:                  standard GELU MLP (bias-free, timm pattern)
  - RMSNorm:              replaces LayerNorm (no mean-centering, fp32)
  - Layer Scale:          learned per-branch scaling (gamma init 1e-4)
  - DropPath:             stochastic depth

Self-contained — no imports from eventfft_arch.py.

Source components:
  - RMSNorm:                reference_2/models_vit5.py:75-86
  - VisionRotaryEmbedding:  reference_2/rope.py:34-70 (adapted for windows)
  - Attention pattern:      reference_2/models_vit5.py:17-73 (adapted for windows)
  - Block pattern:          reference_2/models_vit5.py:113-139
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_


# ---------------------------------------------------------------------------
# RMSNorm  (reference_2/models_vit5.py:75-86)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Operates on last dimension. fp32 compute, cast back to input dtype. No bias.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


# ---------------------------------------------------------------------------
# 2D Rotary Position Embedding  (reference_2/rope.py, adapted for windows)
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Standard RoPE rotation: split last dim into pairs, rotate by 90 degrees."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def broadcat(tensors, dim=-1):
    """Broadcast tensors to compatible shapes then concatenate (reference_2/rope.py:12-26)."""
    shapes = [list(t.shape) for t in tensors]
    ndim = len(shapes[0])
    dim = dim % ndim
    for i in range(ndim):
        if i == dim:
            continue
        max_size = max(s[i] for s in shapes)
        for s in shapes:
            s[i] = max_size
    tensors = [t.expand(*s) for t, s in zip(tensors, shapes)]
    return torch.cat(tensors, dim=dim)


class VisionRotaryEmbedding2d(nn.Module):
    """2D Rotary Position Embedding with caching.

    Operates on (*, N, num_heads, head_dim) format. Builds a 2D frequency grid
    from (H, W) and caches cos/sin tensors.

    For window attention: H, W are the window dimensions (win_size, win_size),
    so positions encode location within each window.

    Args:
        dim: Half of head_dim. Row freqs cover dim channels, column freqs
             cover dim channels, concatenated to 2*dim = head_dim.
        theta: RoPE base frequency (default 10000).
    """

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self._cache_cos = None
        self._cache_sin = None
        self._cache_key = None

    def _build_cache(self, H, W, device):
        freqs = 1.0 / (self.theta ** (
            torch.arange(0, self.dim, 2, device=device)[:(self.dim // 2)].float()
            / self.dim))

        t_h = torch.arange(H, device=device, dtype=torch.float32)
        t_w = torch.arange(W, device=device, dtype=torch.float32)

        freqs_h = torch.einsum('i, f -> i f', t_h, freqs)  # (H, dim//2)
        freqs_w = torch.einsum('i, f -> i f', t_w, freqs)  # (W, dim//2)

        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)  # (H, dim)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)  # (W, dim)

        # 2D grid: (H, W, 2*dim = head_dim)
        grid = broadcat([freqs_h[:, None, :], freqs_w[None, :, :]], dim=-1)
        grid = rearrange(grid, 'h w d -> (h w) d')  # (H*W, head_dim)

        self._cache_cos = grid.cos()
        self._cache_sin = grid.sin()
        self._cache_key = (H, W, device)

    def forward(self, x, H, W):
        """Apply 2D RoPE.

        Args:
            x: (*, N, num_heads, head_dim) where N = H * W.
            H, W: spatial grid dimensions.
        Returns:
            Same shape with RoPE applied.
        """
        if self._cache_key != (H, W, x.device):
            self._build_cache(H, W, x.device)

        # Broadcast: (N, head_dim) -> (1, N, 1, head_dim) for (B, N, heads, hd)
        cos = self._cache_cos[None, :, None, :]
        sin = self._cache_sin[None, :, None, :]

        return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Window Partition / Reverse
# ---------------------------------------------------------------------------

def window_partition(x, H, W, win_size):
    """Partition tokens into non-overlapping windows.

    Args:
        x: (B, H*W, C) token tensor.
        H, W: spatial dimensions.
        win_size: window size (H, W must be divisible by win_size).
    Returns:
        (B * num_windows, win_size*win_size, C)
    """
    x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
    x = rearrange(x, 'b (nH wH) (nW wW) c -> (b nH nW) (wH wW) c',
                  wH=win_size, wW=win_size)
    return x


def window_reverse(x, H, W, win_size):
    """Reverse window partition back to token sequence.

    Args:
        x: (B * num_windows, win_size*win_size, C)
        H, W: spatial dimensions.
        win_size: window size.
    Returns:
        (B, H*W, C)
    """
    nH, nW = H // win_size, W // win_size
    x = rearrange(x, '(b nH nW) (wH wW) c -> b (nH wH) (nW wW) c',
                  nH=nH, nW=nW, wH=win_size, wW=win_size)
    return rearrange(x, 'b h w c -> b (h w) c')


# ---------------------------------------------------------------------------
# Mlp  (timm standard pattern, bias-free)
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """Standard 2-layer GELU MLP (bias-free).

    Args:
        dim: Input/output dimension.
        mlp_ratio: Hidden dimension = dim * mlp_ratio.
        drop: Dropout rate.
    """

    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ---------------------------------------------------------------------------
# WindowSelfAttention
# ---------------------------------------------------------------------------

class WindowSelfAttention(nn.Module):
    """Windowed multi-head self-attention with QK Norm + 2D RoPE.

    Based on reference_2/models_vit5.py Attention class, adapted:
      - Windowed: partitions spatial tokens into win_size x win_size windows
      - 2D RoPE encodes position within each window
      - QK Norm: RMSNorm on q and k per head independently
      - All projections bias-free

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        win_size: Window size for spatial partitioning.
        qk_norm: Enable QK normalization.
        rope_theta: RoPE base frequency.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
    """

    def __init__(self, dim, num_heads, win_size=8, qk_norm=True,
                 rope_theta=10000, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.win_size = win_size

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rope = VisionRotaryEmbedding2d(self.head_dim // 2, rope_theta)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C) token tensor.
            H, W: spatial dimensions.
        Returns:
            (B, H*W, C)
        """
        B, N, C = x.shape
        ws = self.win_size

        # Window partition
        x_win = window_partition(x, H, W, ws)  # (B*nW, ws*ws, C)
        Bw = x_win.shape[0]

        # QKV projection
        qkv = self.qkv(x_win).reshape(Bw, ws * ws, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (Bw, ws*ws, heads, head_dim)

        # QK Norm
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 2D RoPE within each window
        q = self.rope(q, ws, ws)
        k = self.rope(k, ws, ws)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # (Bw, heads, ws*ws, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_win = (attn @ v).transpose(1, 2).reshape(Bw, ws * ws, C)
        x_win = self.proj(x_win)
        x_win = self.proj_drop(x_win)

        # Window reverse
        return window_reverse(x_win, H, W, ws)


# ---------------------------------------------------------------------------
# WindowCrossAttention
# ---------------------------------------------------------------------------

class WindowCrossAttention(nn.Module):
    """Windowed multi-head cross-attention: image Q x event KV.

    Image features supply Q; event features supply K and V.
    No RoPE (cross-modal positions are already spatially aligned).

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        win_size: Window size for spatial partitioning.
        qk_norm: Enable QK normalization.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
    """

    def __init__(self, dim, num_heads, win_size=8, qk_norm=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.win_size = win_size

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, evt, H, W):
        """
        Args:
            x: image tokens (B, N, C).
            evt: event tokens (B, N, C).
            H, W: spatial dimensions.
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        ws = self.win_size

        # Window partition both streams
        x_win = window_partition(x, H, W, ws)     # (B*nW, ws*ws, C)
        evt_win = window_partition(evt, H, W, ws)
        Bw = x_win.shape[0]

        # Q from image, KV from event
        q = self.q_proj(x_win).reshape(Bw, ws * ws, self.num_heads, self.head_dim)
        kv = self.kv_proj(evt_win).reshape(Bw, ws * ws, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        # QK Norm
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scaled dot-product attention (no RoPE for cross-attention)
        q = q.transpose(1, 2)  # (Bw, heads, ws*ws, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_win = (attn @ v).transpose(1, 2).reshape(Bw, ws * ws, C)
        x_win = self.proj(x_win)
        x_win = self.proj_drop(x_win)

        return window_reverse(x_win, H, W, ws)


# ---------------------------------------------------------------------------
# ImageBlock — core building block for image stream
# ---------------------------------------------------------------------------

class ImageBlock(nn.Module):
    """ViT-5 style transformer block for spatial features.

    All computation in token format (B, N, C) internally. Spatial reshape at
    block boundary only.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim = dim * mlp_ratio.
        drop_path: Drop path rate.
        qk_norm: Enable QK Norm in attention.
        rope_theta: RoPE base frequency.
        win_size: Window attention size.
        layer_scale: Enable Layer Scale.
        layer_scale_init: Initial gamma value.
        att: If True, enable self-attention branch (decoder/refinement).
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0.,
                 qk_norm=True, rope_theta=10000, win_size=8,
                 layer_scale=True, layer_scale_init=1e-4, att=False):
        super().__init__()
        self.att = att

        if att:
            self.norm1 = RMSNorm(dim)
            self.attn = WindowSelfAttention(
                dim, num_heads, win_size, qk_norm, rope_theta)

        self.norm2 = RMSNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_scale = layer_scale
        if layer_scale:
            if att:
                self.gamma_1 = nn.Parameter(
                    layer_scale_init * torch.ones(dim))
            self.gamma_2 = nn.Parameter(
                layer_scale_init * torch.ones(dim))

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        if self.att:
            residual = self.attn(self.norm1(x), H, W)
            if self.layer_scale:
                x = x + self.drop_path(self.gamma_1 * residual)
            else:
                x = x + self.drop_path(residual)

        residual = self.mlp(self.norm2(x))
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_2 * residual)
        else:
            x = x + self.drop_path(residual)

        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)


# ---------------------------------------------------------------------------
# EventBlock — lightweight block for event stream (no attention)
# ---------------------------------------------------------------------------

class EventBlock(nn.Module):
    """Lightweight transformer block for event stream: RMSNorm + Mlp only.

    No attention — keeps event branch lightweight. Same Mlp as image stream
    for consistency.

    Args:
        dim: Feature dimension.
        mlp_ratio: MLP hidden dim = dim * mlp_ratio.
        drop_path: Drop path rate.
        layer_scale: Enable Layer Scale.
        layer_scale_init: Initial gamma value.
    """

    def __init__(self, dim, mlp_ratio=4., drop_path=0.,
                 layer_scale=True, layer_scale_init=1e-4):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        residual = self.mlp(self.norm(x))
        if self.layer_scale:
            x = x + self.drop_path(self.gamma * residual)
        else:
            x = x + self.drop_path(residual)

        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)


# ---------------------------------------------------------------------------
# CrossModalFuse — cross-modal fusion with windowed cross-attention
# ---------------------------------------------------------------------------

class CrossModalFuse(nn.Module):
    """Fuses image + event via windowed cross-attention followed by Mlp.

    Image features supply Q; event features supply K and V.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim = dim * mlp_ratio.
        drop_path: Drop path rate.
        qk_norm: Enable QK Norm in cross-attention.
        win_size: Window size.
        layer_scale: Enable Layer Scale.
        layer_scale_init: Initial gamma value.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0.,
                 qk_norm=True, win_size=8,
                 layer_scale=True, layer_scale_init=1e-4):
        super().__init__()
        self.norm_img = RMSNorm(dim)
        self.norm_evt = RMSNorm(dim)
        self.attn = WindowCrossAttention(dim, num_heads, win_size, qk_norm)

        self.norm_ffn = RMSNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x, evt):
        """x, evt: (B, C, H, W) -> x: (B, C, H, W)"""
        B, C, H, W = x.shape
        x_tok = rearrange(x, 'b c h w -> b (h w) c')
        evt_tok = rearrange(evt, 'b c h w -> b (h w) c')

        residual = self.attn(self.norm_img(x_tok), self.norm_evt(evt_tok), H, W)
        if self.layer_scale:
            x_tok = x_tok + self.drop_path(self.gamma_1 * residual)
        else:
            x_tok = x_tok + self.drop_path(residual)

        residual = self.mlp(self.norm_ffn(x_tok))
        if self.layer_scale:
            x_tok = x_tok + self.drop_path(self.gamma_2 * residual)
        else:
            x_tok = x_tok + self.drop_path(residual)

        return rearrange(x_tok, 'b (h w) c -> b c h w', h=H, w=W)


# ---------------------------------------------------------------------------
# SkipFuse — skip-connection fusion for decoder
# ---------------------------------------------------------------------------

class SkipFuse(nn.Module):
    """Fuses encoder skip-connection with decoder features.

    Concat along channel dim -> Conv1x1 -> ImageBlock(att=True) -> Conv1x1
    -> split + add.  Returns a single tensor (same as original Fuse).

    Args:
        dim: Feature dimension (per branch, concatenated = 2*dim).
        num_heads: Attention heads for the internal ImageBlock.
        mlp_ratio: MLP ratio.
        drop_path: Drop path rate.
        qk_norm: QK Norm toggle.
        rope_theta: RoPE base frequency.
        win_size: Window size.
        layer_scale: Layer Scale toggle.
        layer_scale_init: Layer Scale init value.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0.,
                 qk_norm=True, rope_theta=10000, win_size=8,
                 layer_scale=True, layer_scale_init=1e-4):
        super().__init__()
        self.n_feat = dim
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)
        self.block = ImageBlock(
            dim * 2, num_heads * 2, mlp_ratio, drop_path,
            qk_norm=qk_norm, rope_theta=rope_theta, win_size=win_size,
            layer_scale=layer_scale, layer_scale_init=layer_scale_init,
            att=True)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)

    def forward(self, dec, enc):
        """dec, enc: (B, C, H, W) -> (B, C, H, W)"""
        x = self.conv1(torch.cat([enc, dec], dim=1))
        x = self.block(x)
        x = self.conv2(x)
        e, d = x.chunk(2, dim=1)
        return e + d


# ---------------------------------------------------------------------------
# Structural modules (self-contained copies from eventfft_arch)
# ---------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    """3x3 Conv embedding, preserves spatial resolution."""

    def __init__(self, in_c=3, embed_dim=48):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    """Bilinear 0.5x + Conv to double channels."""

    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(dim, dim * 2, 3, 1, 1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Bilinear 2x + Conv to halve channels."""

    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim, dim // 2, 3, 1, 1, bias=False))

    def forward(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------
# EventFFTViT5 — main model
# ---------------------------------------------------------------------------

class EventFFTViT5(nn.Module):
    """Event-guided image restoration with pure ViT-5 transformer components.

    Dual-stream 3-level U-Net:
      - Event encoder:  EventBlock (lightweight, no attention)
      - Image encoder:  ImageBlock (att=False, FFN only)
      - Cross-modal fusion: CrossModalFuse (windowed cross-attention)
      - Decoder:        ImageBlock (att=True, window self-attention + FFN)
      - Refinement:     ImageBlock (att=True)
      - Skip fuse:      SkipFuse (concat + ImageBlock + split)

    Args:
        evt_inp: Event input channels (default 6).
        img_inp: Image input channels (default 3).
        out_channels: Output channels (default 3).
        dim: Base feature dimension (default 48).
        num_blocks: Blocks per level [L1, L2, L3].
        num_refinement_blocks: Refinement blocks after decoder.
        num_heads: Attention heads per level [L1, L2, L3].
        mlp_ratio: MLP hidden dim = dim * mlp_ratio.
        win_size: Window attention size.
        drop_path_rate: Stochastic depth rate.
        bias: Bias for the output Conv only.
        qk_norm: Enable QK Norm in attention.
        rope_theta: RoPE base frequency.
        layer_scale: Enable Layer Scale.
        layer_scale_init: Initial gamma value for Layer Scale.
    """

    def __init__(
        self,
        evt_inp=6,
        img_inp=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 4, 8),
        num_refinement_blocks=4,
        num_heads=(1, 2, 4),
        mlp_ratio=4.,
        win_size=8,
        drop_path_rate=0.,
        bias=False,
        qk_norm=True,
        rope_theta=10000,
        layer_scale=True,
        layer_scale_init=1e-4,
    ):
        super().__init__()

        dp = drop_path_rate
        ls_kw = dict(layer_scale=layer_scale, layer_scale_init=layer_scale_init)
        attn_kw = dict(qk_norm=qk_norm, rope_theta=rope_theta, win_size=win_size)

        # ---- Embeddings ----
        self.evt_embed = OverlapPatchEmbed(evt_inp, dim)
        self.img_embed = OverlapPatchEmbed(img_inp, dim)

        # ---- Event encoder (lightweight: EventBlock, no attention) ----
        self.evt_encoder_level1 = nn.Sequential(*[
            EventBlock(dim, mlp_ratio, dp, **ls_kw)
            for _ in range(num_blocks[0])])
        self.evt_down1_2 = Downsample(dim)

        self.evt_encoder_level2 = nn.Sequential(*[
            EventBlock(dim * 2, mlp_ratio, dp, **ls_kw)
            for _ in range(num_blocks[1])])
        self.evt_down2_3 = Downsample(dim * 2)

        self.evt_encoder_level3 = nn.Sequential(*[
            EventBlock(dim * 4, mlp_ratio, dp, **ls_kw)
            for _ in range(num_blocks[2])])

        # ---- Image encoder (ImageBlock, att=False) ----
        self.encoder_level1 = nn.Sequential(*[
            ImageBlock(dim, num_heads[0], mlp_ratio, dp,
                       att=False, **attn_kw, **ls_kw)
            for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            ImageBlock(dim * 2, num_heads[1], mlp_ratio, dp,
                       att=False, **attn_kw, **ls_kw)
            for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(*[
            ImageBlock(dim * 4, num_heads[2], mlp_ratio, dp,
                       att=False, **attn_kw, **ls_kw)
            for _ in range(num_blocks[2])])

        # ---- Cross-modal fusion ----
        fuse_kw = dict(qk_norm=qk_norm, win_size=win_size, **ls_kw)
        self.fuse_en1 = CrossModalFuse(
            dim, num_heads[0], mlp_ratio, dp, **fuse_kw)
        self.fuse_en2 = CrossModalFuse(
            dim * 2, num_heads[1], mlp_ratio, dp, **fuse_kw)
        self.fuse_en3 = CrossModalFuse(
            dim * 4, num_heads[2], mlp_ratio, dp, **fuse_kw)

        # ---- Decoder (ImageBlock, att=True) ----
        self.decoder_level3 = nn.Sequential(*[
            ImageBlock(dim * 4, num_heads[2], mlp_ratio, dp,
                       att=True, **attn_kw, **ls_kw)
            for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(dim * 4)
        self.decoder_level2 = nn.Sequential(*[
            ImageBlock(dim * 2, num_heads[1], mlp_ratio, dp,
                       att=True, **attn_kw, **ls_kw)
            for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            ImageBlock(dim, num_heads[0], mlp_ratio, dp,
                       att=True, **attn_kw, **ls_kw)
            for _ in range(num_blocks[0])])

        # ---- Refinement (ImageBlock, att=True) ----
        self.refinement = nn.Sequential(*[
            ImageBlock(dim, num_heads[0], mlp_ratio, dp,
                       att=True, **attn_kw, **ls_kw)
            for _ in range(num_refinement_blocks)])

        # ---- Skip-connection fuse ----
        skip_kw = dict(qk_norm=qk_norm, rope_theta=rope_theta,
                       win_size=win_size, **ls_kw)
        self.fuse2 = SkipFuse(dim * 2, num_heads[1], mlp_ratio, dp, **skip_kw)
        self.fuse1 = SkipFuse(dim, num_heads[0], mlp_ratio, dp, **skip_kw)

        # ---- Output ----
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

        # ---- Weight initialization ----
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, event):
        """
        Args:
            x:     (B, img_inp, H, W) — blurry / noisy image.
            event: (B, evt_inp, H, W) — event voxel grid.
        Returns:
            (B, out_channels, H, W)
        """
        # ---- Event encoder ----
        evt_inp = self.evt_embed(event)
        img_inp = self.img_embed(x)

        evt_out_enc_level1 = self.evt_encoder_level1(evt_inp)

        evt_inp_enc_level2 = self.evt_down1_2(evt_out_enc_level1)
        evt_out_enc_level2 = self.evt_encoder_level2(evt_inp_enc_level2)

        evt_inp_enc_level3 = self.evt_down2_3(evt_out_enc_level2)
        evt_out_enc_level3 = self.evt_encoder_level3(evt_inp_enc_level3)

        # ---- Image encoder + cross-modal fusion ----
        out_enc_level1 = self.encoder_level1(img_inp)
        out_enc_level1 = self.fuse_en1(out_enc_level1, evt_out_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2 = self.fuse_en2(out_enc_level2, evt_out_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc_level3 = self.fuse_en3(out_enc_level3, evt_out_enc_level3)

        # ---- Decoder ----
        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # ---- Refinement + output + global residual ----
        out_dec_level1 = self.refinement(out_dec_level1)
        return self.output(out_dec_level1) + x
