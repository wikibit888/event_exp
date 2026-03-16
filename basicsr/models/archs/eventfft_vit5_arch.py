"""EventFFTViT5 — Dual-stream event-guided image restoration with ViT-5 enhancements.

Integrates modernized ViT-5 (reference_2, arXiv 2026-02) components into the
eventfft_cross dual-stream U-Net architecture:
  - RMSNorm:    replaces LayerNorm (no mean-centering, fp32 numerics)
  - 2D RoPE:    spatial position encoding via rotary embeddings on q/k
  - QK Norm:    RMSNorm applied to q and k before FFT correlation
  - Layer Scale: learned per-branch scaling (gamma init 1e-4)

Fixes the cross-modal fusion bug in evfft_cross_arch.py where fuse_en1/2/3
(Event_Image_Fuse) are overwritten by Fuse.

Source components:
  - RMSNorm:                  reference_2/models_vit5.py:75-86
  - rotate_half, broadcat:    reference_2/rope.py:12-32
  - VisionRotaryEmbedding:    reference_2/rope.py:34-70 (adapted for (B,C,H,W))
  - Layer Scale pattern:      reference_2/models_vit5.py:127-138
  - QK Norm pattern:          reference_2/models_vit5.py:48-51
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from basicsr.models.archs.eventfft_arch import (
    DFFN,
    Event_Transformer_Block,
    Fuse,
    OverlapPatchEmbed,
    Downsample,
    Upsample,
)


# ---------------------------------------------------------------------------
# RMSNorm  (reference_2/models_vit5.py:75-86)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes by RMS only (no mean-centering), then scales with learned weight.
    Numerically computed in float32, output cast back to input dtype. No bias.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RMSNorm2d(nn.Module):
    """Spatial wrapper: applies RMSNorm to (B, C, H, W) feature maps.

    Same pattern as LayerNorm wrapper in eventfft_arch.py:56-67.
    (B, C, H, W) -> rearrange to (B, H*W, C) -> RMSNorm -> reshape back.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.body = RMSNorm(dim, eps)

    def forward(self, x):
        h, w = x.shape[-2:]
        # (B, C, H, W) -> (B, H*W, C)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        # (B, H*W, C) -> (B, C, H, W)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# ---------------------------------------------------------------------------
# 2D Rotary Position Embedding  (reference_2/rope.py:12-70, adapted)
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Standard RoPE rotation helper (reference_2/rope.py:28-32).

    Splits last dimension into pairs, rotates each pair by 90 degrees.
    Requires last dimension to be even.
    """
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def broadcat(freqss, dim=-1):
    """Broadcast tensors to compatible shapes then concatenate (reference_2/rope.py:12-26)."""
    num_freqss = len(freqss)
    shape_lens = set(list(map(lambda t: len(t.shape), freqss)))
    assert len(shape_lens) == 1, 'freqss must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), freqss)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), \
        'invalid dimensions for broadcastable concatenation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_freqss), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    freqss = list(map(lambda t: t[0].expand(*t[1]), zip(freqss, expandable_shapes)))
    return torch.cat(freqss, dim=dim)


class VisionRotaryEmbedding2d(nn.Module):
    """2D Rotary Position Embedding for spatial feature maps.

    Adapted from reference_2/rope.py:34-70 (VisionRotaryEmbedding).

    Changes vs original:
      - Accepts (B, C, H, W) input instead of (N, heads, head_dim)
      - Uses raw position indices (no pt_seq_len normalization — restoration
        has no fixed pretrain resolution)
      - Device-safe: no .cuda() calls, uses x.device
      - Dynamically generates freq grid from input H, W

    Args:
        dim: Half of the target channel count. Row freqs cover dim channels,
             column freqs cover dim channels, concatenated to 2*dim total.
        theta: RoPE base frequency (default 10000).
    """

    def __init__(self, dim, theta=10000):
        super().__init__()
        # Store hyperparams only — no register_buffer.
        # Avoids DDP buffer broadcast deadlock during rank-0-only validation.
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W) with 2D RoPE applied."""
        B, C, H, W = x.shape
        device = x.device

        # Recompute freq base inline (dim//2 scalars, negligible cost)
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device)[:(self.dim // 2)].float() / self.dim))

        # Work in (B, H, W, C) so last dim is channels for rotate_half
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Position indices for rows and columns
        t_h = torch.arange(H, device=device, dtype=torch.float32)
        t_w = torch.arange(W, device=device, dtype=torch.float32)

        # Per-axis frequencies: outer product of positions and freq bases
        freqs_h = torch.einsum('i, f -> i f', t_h, freqs)  # (H, dim//2)
        freqs_w = torch.einsum('i, f -> i f', t_w, freqs)  # (W, dim//2)

        # Repeat for cos/sin pair interleaving
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)  # (H, dim)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)  # (W, dim)

        # 2D grid via broadcat: (H,1,dim) + (1,W,dim) -> (H, W, 2*dim=C)
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        x = x * freqs.cos() + rotate_half(x) * freqs.sin()

        return x.permute(0, 3, 1, 2)  # back to (B, C, H, W)


# ---------------------------------------------------------------------------
# FSAS_V5 — FSAS with QK Norm + RoPE + RMSNorm
# ---------------------------------------------------------------------------

class FSAS_V5(nn.Module):
    """Frequency-domain Self-Attention Style module with ViT-5 enhancements.

    Based on FSAS (eventfft_arch.py:125-176) with:
      - RMSNorm2d replaces LayerNorm for correlation normalization
      - QK Norm: RMSNorm2d on q and k independently (reference_2 pattern)
      - 2D RoPE: spatial position encoding on q/k before FFT correlation

    RoPE + FFT rationale: RoPE encodes spatial position into q/k features
    *before* patch-wise FFT correlation. The position info participates in the
    frequency-domain matching, giving patches position awareness.
    """

    def __init__(self, dim, bias, qk_norm=True, use_rope=True, rope_theta=10000):
        super().__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(
            dim * 6, dim * 6, kernel_size=3, stride=1, padding=1,
            groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = RMSNorm2d(dim * 2)

        self.patch_size = 8

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm2d(dim * 2)
            self.k_norm = RMSNorm2d(dim * 2)

        self.use_rope = use_rope
        if use_rope:
            self.rope = VisionRotaryEmbedding2d(dim, rope_theta)

    def forward(self, x):
        # x: (B, dim, H, W)
        hidden = self.to_hidden(x)  # (B, 6*dim, H, W)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)  # each (B, 2*dim, H, W)

        # QK Norm: stabilize q/k magnitudes before correlation
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 2D RoPE: encode spatial positions into q/k
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        # Patch-level FFT correlation (same as original FSAS)
        p = self.patch_size
        q_patch = rearrange(q, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p)
        k_patch = rearrange(k, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p)

        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(p, p))

        out = rearrange(out, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=p, p2=p)
        out = self.norm(out)

        output = v * out
        output = self.project_out(output)
        return output


# ---------------------------------------------------------------------------
# TransformerBlock_V5 — TransformerBlock with Layer Scale + RMSNorm + FSAS_V5
# ---------------------------------------------------------------------------

class TransformerBlock_V5(nn.Module):
    """Enhanced TransformerBlock with ViT-5 components.

    Based on TransformerBlock (eventfft_arch.py:286-310) with:
      - RMSNorm2d replaces LayerNorm
      - FSAS_V5 replaces FSAS (QK Norm + RoPE)
      - Layer Scale: learned gamma_1, gamma_2 scalars (init 1e-4)

    Args:
        dim: Feature dimension.
        ffn_expansion_factor: DFFN expansion ratio.
        bias: Conv bias.
        att: If True, enable FSAS_V5 attention branch.
        qk_norm: Enable QK Norm in FSAS_V5.
        use_rope: Enable 2D RoPE in FSAS_V5.
        rope_theta: RoPE base frequency.
        layer_scale: Enable Layer Scale.
        layer_scale_init: Initial value for gamma (default 1e-4).
    """

    def __init__(self, dim, ffn_expansion_factor=2, bias=False, att=False,
                 qk_norm=True, use_rope=True, rope_theta=10000,
                 layer_scale=True, layer_scale_init=1e-4):
        super().__init__()

        self.att = att
        if self.att:
            self.norm1 = RMSNorm2d(dim)
            self.attn = FSAS_V5(dim, bias, qk_norm, use_rope, rope_theta)

        self.norm2 = RMSNorm2d(dim)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

        self.layer_scale = layer_scale
        if layer_scale:
            if self.att:
                self.gamma_1 = nn.Parameter(
                    layer_scale_init * torch.ones(dim))
            self.gamma_2 = nn.Parameter(
                layer_scale_init * torch.ones(dim))

    def forward(self, x):
        if self.att:
            if self.layer_scale:
                x = x + self.gamma_1.view(1, -1, 1, 1) * self.attn(self.norm1(x))
            else:
                x = x + self.attn(self.norm1(x))

        if self.layer_scale:
            x = x + self.gamma_2.view(1, -1, 1, 1) * self.ffn(self.norm2(x))
        else:
            x = x + self.ffn(self.norm2(x))

        return x


# ---------------------------------------------------------------------------
# CrossModalFFTAttn_V5 — cross-modal attention with QK Norm + RMSNorm
# ---------------------------------------------------------------------------

class CrossModalFFTAttn_V5(nn.Module):
    """Frequency-domain cross-modal attention with ViT-5 enhancements.

    Based on CrossModalFFTAttn (eventfft_mamba_arch.py:261-302) with:
      - RMSNorm2d replaces all LayerNorm instances
      - QK Norm on q and k (if enabled)
      - No RoPE (cross-modal: position already aligned between streams)

    Image features supply Q; event features supply K and V.
    Correlation computed via patch-level rfft2 multiplication.
    """

    def __init__(self, dim, bias, qk_norm=True, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.norm_img = RMSNorm2d(dim)
        self.norm_evt = RMSNorm2d(dim)

        self.q = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 4, 1, bias=bias)
        self.q_dw = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=bias)
        self.kv_dw = nn.Conv2d(dim * 4, dim * 4, 3, padding=1, groups=dim * 4, bias=bias)

        self.norm_corr = RMSNorm2d(dim * 2)
        self.proj_out = nn.Conv2d(dim * 2, dim, 1, bias=bias)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm2d(dim * 2)
            self.k_norm = RMSNorm2d(dim * 2)

    def forward(self, x, evt):
        """x, evt: (B, C, H, W)"""
        x = self.norm_img(x)
        evt = self.norm_evt(evt)

        q = self.q_dw(self.q(x))         # (B, 2C, H, W)
        kv = self.kv_dw(self.kv(evt))
        k, v = kv.chunk(2, dim=1)        # each (B, 2C, H, W)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        p = self.patch_size
        q_p = rearrange(q, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p)
        k_p = rearrange(k, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p)

        corr = torch.fft.rfft2(q_p.float()) * torch.fft.rfft2(k_p.float())
        corr = torch.fft.irfft2(corr, s=(p, p))
        corr = rearrange(corr, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=p, p2=p)
        corr = self.norm_corr(corr)

        return self.proj_out(v * corr)   # (B, C, H, W)


# ---------------------------------------------------------------------------
# CrossModalFuse_V5 — cross-modal fusion with Layer Scale
# ---------------------------------------------------------------------------

class CrossModalFuse_V5(nn.Module):
    """Fuses image + event via CrossModalFFTAttn_V5 followed by DFFN, with Layer Scale.

    Based on CrossModalFuse (eventfft_mamba_arch.py:309-321) with:
      - CrossModalFFTAttn_V5 (QK Norm + RMSNorm)
      - Layer Scale on both attention and FFN branches
    """

    def __init__(self, dim, ffn_expansion_factor=2, bias=False,
                 qk_norm=True, layer_scale=True, layer_scale_init=1e-4):
        super().__init__()
        self.attn = CrossModalFFTAttn_V5(dim, bias, qk_norm)
        self.norm = RMSNorm2d(dim)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x, evt):
        if self.layer_scale:
            x = x + self.gamma_1.view(1, -1, 1, 1) * self.attn(x, evt)
            x = x + self.gamma_2.view(1, -1, 1, 1) * self.ffn(self.norm(x))
        else:
            x = x + self.attn(x, evt)
            x = x + self.ffn(self.norm(x))
        return x


# ---------------------------------------------------------------------------
# EventFFTViT5 — main model
# ---------------------------------------------------------------------------

class EventFFTViT5(nn.Module):
    """Event-guided image restoration with ViT-5 modernized components.

    Same dual-stream U-Net structure as eventfft_cross (evfft_cross_arch.py)
    but with V5-enhanced blocks and correct cross-modal fusion.

    Substitutions vs eventfft_cross:
      - Image encoder blocks:   TransformerBlock(att=False) -> TransformerBlock_V5(att=False)
      - Decoder blocks:         TransformerBlock(att=True)  -> TransformerBlock_V5(att=True)
      - Refinement blocks:      TransformerBlock(att=True)  -> TransformerBlock_V5(att=True)
      - Cross-modal fusion:     Fuse (buggy overwrite)      -> CrossModalFuse_V5 (correct)
      - Event encoder:          Event_Transformer_Block      (unchanged)
      - Skip-connection fuse:   Fuse                         (unchanged)
      - Embed/Down/Up:          same                         (imported)

    Args:
        evt_inp: Event input channels (default 6).
        img_inp: Image input channels (default 3).
        out_channels: Output channels (default 3).
        dim: Base feature dimension (default 48).
        num_blocks: Blocks per encoder level [L1, L2, L3].
        num_refinement_blocks: Refinement blocks after decoder.
        ffn_expansion_factor: DFFN expansion ratio.
        bias: Conv bias.
        qk_norm: Enable QK Norm in V5 attention blocks.
        use_rope: Enable 2D RoPE in FSAS_V5.
        rope_theta: RoPE base frequency.
        layer_scale: Enable Layer Scale.
        layer_scale_init: Initial value for layer scale gamma.
    """

    def __init__(
        self,
        evt_inp=6,
        img_inp=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 4, 8),
        num_refinement_blocks=4,
        ffn_expansion_factor=3,
        bias=False,
        qk_norm=True,
        use_rope=True,
        rope_theta=10000,
        layer_scale=True,
        layer_scale_init=1e-4,
    ):
        super().__init__()

        # ---- embeddings ----
        self.evt_embed = OverlapPatchEmbed(evt_inp, dim)
        self.img_embed = OverlapPatchEmbed(img_inp, dim)

        # ---- event encoder (unchanged: FFN only) ----
        self.evt_encoder_level1 = nn.Sequential(*[
            Event_Transformer_Block(
                dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[0])])
        self.evt_down1_2 = Downsample(dim)

        self.evt_encoder_level2 = nn.Sequential(*[
            Event_Transformer_Block(
                dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[1])])
        self.evt_down2_3 = Downsample(dim * 2)

        self.evt_encoder_level3 = nn.Sequential(*[
            Event_Transformer_Block(
                dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[2])])

        # V5 block kwargs shared across encoder/decoder/refinement
        v5_kw = dict(
            qk_norm=qk_norm, use_rope=use_rope, rope_theta=rope_theta,
            layer_scale=layer_scale, layer_scale_init=layer_scale_init,
        )

        # ---- image encoder (V5 blocks, att=False) ----
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock_V5(
                dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, att=False, **v5_kw)
            for _ in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock_V5(
                dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, att=False, **v5_kw)
            for _ in range(num_blocks[1])])

        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock_V5(
                dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, att=False, **v5_kw)
            for _ in range(num_blocks[2])])

        # ---- cross-modal fusion (V5, fixes overwrite bug) ----
        fuse_kw = dict(
            qk_norm=qk_norm, layer_scale=layer_scale,
            layer_scale_init=layer_scale_init,
        )
        self.fuse_en1 = CrossModalFuse_V5(
            dim, ffn_expansion_factor, bias, **fuse_kw)
        self.fuse_en2 = CrossModalFuse_V5(
            dim * 2, ffn_expansion_factor, bias, **fuse_kw)
        self.fuse_en3 = CrossModalFuse_V5(
            dim * 4, ffn_expansion_factor, bias, **fuse_kw)

        # ---- decoder (V5 blocks, att=True) ----
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock_V5(
                dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, att=True, **v5_kw)
            for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(dim * 4)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock_V5(
                dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, att=True, **v5_kw)
            for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock_V5(
                dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, att=True, **v5_kw)
            for _ in range(num_blocks[0])])

        # ---- refinement (V5 blocks, att=True) ----
        self.refinement = nn.Sequential(*[
            TransformerBlock_V5(
                dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, att=True, **v5_kw)
            for _ in range(num_refinement_blocks)])

        # ---- skip-connection fuse (unchanged) ----
        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)

        self.output = nn.Conv2d(
            dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, event):
        """
        Args:
            x:     (B, img_inp, H, W) — blurry / noisy image.
            event: (B, evt_inp, H, W) — event voxel grid.
        Returns:
            (B, out_channels, H, W)
        """
        # ---- event encoder ----
        evt_inp = self.evt_embed(event)
        img_inp = self.img_embed(x)

        evt_out_enc_level1 = self.evt_encoder_level1(evt_inp)

        evt_inp_enc_level2 = self.evt_down1_2(evt_out_enc_level1)
        evt_out_enc_level2 = self.evt_encoder_level2(evt_inp_enc_level2)

        evt_inp_enc_level3 = self.evt_down2_3(evt_out_enc_level2)
        evt_out_enc_level3 = self.evt_encoder_level3(evt_inp_enc_level3)

        # ---- image encoder + cross-modal fusion ----
        out_enc_level1 = self.encoder_level1(img_inp)
        out_enc_level1 = self.fuse_en1(out_enc_level1, evt_out_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2 = self.fuse_en2(out_enc_level2, evt_out_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc_level3 = self.fuse_en3(out_enc_level3, evt_out_enc_level3)

        # ---- decoder ----
        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # ---- refinement ----
        out_dec_level1 = self.refinement(out_dec_level1)

        # ---- output + global residual ----
        return self.output(out_dec_level1) + x
