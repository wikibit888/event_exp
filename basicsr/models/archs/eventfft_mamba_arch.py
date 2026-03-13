"""EventFFTMamba — Dual-stream event-guided image restoration network.

Combines:
  - MaIR (reference_1, CVPR 2025): NSS scanning strategy + SSA sequence aggregation
  - eventfft_cross (event_exp): FFT frequency-domain attention and FFN

Architecture:
  - Event encoder:  Event_Transformer_Block (patch-level FFT FFN)
  - Image encoder:  MambaFFTBlock (NSS Mamba SSM + DFFN FFT FFN)
  - Cross-modal:    CrossModalFFTAttn (freq-domain cross-attention, image Q / event KV)
  - Decoder:        MambaFFTBlock + Fuse skip connections
  - Refinement:     MambaFFTBlock
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import DropPath, to_2tuple
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from basicsr.models.archs.eventfft_arch import (
    LayerNorm,
    DFFN,
    Event_Transformer_Block,
    Fuse,
    OverlapPatchEmbed,
    Downsample,
    Upsample,
)
from basicsr.models.archs.shift_scanf_util import (
    mair_ids_generate,
    mair_ids_scan,
    mair_ids_inverse,
)


# ---------------------------------------------------------------------------
# SSA: Sequence Shuffle Attention  (reference_1 — ShuffleAttn)
# ---------------------------------------------------------------------------

class ShuffleAttn(nn.Module):
    """Channel-shuffle gating to aggregate 4-direction SSM outputs (SSA in MaIR)."""

    def __init__(self, in_features, out_features, group=4):
        super().__init__()
        self.group = group
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=group, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def _channel_shuffle(self, x):
        B, C, H, W = x.shape
        gc = C // self.group
        x = x.reshape(B, gc, self.group, H, W).permute(0, 2, 1, 3, 4)
        return x.reshape(B, C, H, W)

    def _channel_rearrange(self, x):
        B, C, H, W = x.shape
        gc = C // self.group
        x = x.reshape(B, self.group, gc, H, W).permute(0, 2, 1, 3, 4)
        return x.reshape(B, C, H, W)

    def forward(self, x):
        x = self._channel_shuffle(x)
        x = self.gating(x)
        x = self._channel_rearrange(x)
        return x


# ---------------------------------------------------------------------------
# NSS_SSM: Visual Mamba Module with NSS scanning  (reference_1 — LoSh2D)
# ---------------------------------------------------------------------------

class NSS_SSM(nn.Module):
    """
    4-direction SSM with Nested S-shaped Scanning and SSA aggregation.
    Adapted from LoSh2D in reference_1/realDenoising/basicsr/models/archs/mairunet_arch.py.

    Input / output format: (B, H, W, C)  — matches event_exp's internal convention
    when called from MambaFFTBlock which handles the (B,C,H,W) ↔ (B,H,W,C) conversion.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        ssm_ratio=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner,
            groups=self.d_inner, kernel_size=d_conv,
            padding=(d_conv - 1) // 2, bias=conv_bias,
        )
        self.act = nn.SiLU()

        # Shared projection weights for 4 scan directions
        x_proj = [
            nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj]))
        del x_proj

        dt_projs = [self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init,
                                  dt_min, dt_max, dt_init_floor) for _ in range(4)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs]))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs]))
        del dt_projs

        self.A_logs = self._A_log_init(d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.gating = ShuffleAttn(
            in_features=self.d_inner * 4,
            out_features=self.d_inner * 4,
            group=self.d_inner,
        )

    # ------------------------------------------------------------------
    # Parameter init helpers  (identical to reference_1)
    # ------------------------------------------------------------------

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n", d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, merge=True):
        D = torch.ones(d_inner)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_core(self, x, losh_ids):
        """x: (B, d_inner, H, W)  →  returns (B, d_inner*4, H, W)"""
        B, C, H, W = x.shape
        L = H * W
        K = 4

        xs_scan_ids, xs_inverse_ids = losh_ids
        xs = mair_ids_scan(x, xs_scan_ids)          # (B, 4, C*L)

        x_dbl = F.conv1d(
            xs.reshape(B, -1, L),
            self.x_proj_weight.reshape(-1, C, 1),
            groups=K,
        )
        dts, Bs, Cs = torch.split(
            x_dbl.reshape(B, K, -1, L),
            [self.dt_rank, self.d_state, self.d_state], dim=2,
        )
        dts = F.conv1d(
            dts.reshape(B, -1, L),
            self.dt_projs_weight.reshape(K * C, -1, 1),
            groups=K,
        )

        xs  = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs  = Bs.float().view(B, K, -1, L)
        Cs  = Cs.float().view(B, K, -1, L)
        Ds  = self.Ds.float().view(-1)
        As  = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds,
            z=None, delta_bias=dt_bias,
            delta_softplus=True, return_last_state=False,
        ).view(B, K, -1, L)

        return mair_ids_inverse(out_y, xs_inverse_ids, shape=(B, -1, H, W))

    def forward(self, x, losh_ids):
        """x: (B, H, W, C)  →  (B, H, W, C)"""
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)                                # (B,H,W,d_inner) each

        x_in = x_in.permute(0, 3, 1, 2).contiguous()
        x_in = self.act(self.conv2d(x_in))                           # (B, d_inner, H, W)
        y = self._forward_core(x_in, losh_ids)                       # (B, d_inner*4, H, W)

        y = y * self.gating(y)
        y = sum(torch.chunk(y, 4, dim=1))                            # (B, d_inner, H, W)
        y = self.out_norm(y.permute(0, 2, 3, 1))                     # (B, H, W, d_inner)
        y = y * F.silu(z)
        out = self.out_proj(y)                                        # (B, H, W, C)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# CrossModalFFTAttn  (fixed EventFSAS: image Q, event KV, patch-level rfft2)
# ---------------------------------------------------------------------------

class CrossModalFFTAttn(nn.Module):
    """
    Frequency-domain cross-modal attention.
    Image features supply Q; event features supply K and V.
    Correlation computed via patch-level rfft2 multiplication (O(N log N)).

    Fixes the NameError bug present in EventFSAS in eventfft_arch.py.
    """

    def __init__(self, dim, bias, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.norm_img = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm_evt = LayerNorm(dim, LayerNorm_type='WithBias')

        self.q    = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.kv   = nn.Conv2d(dim, dim * 4, 1, bias=bias)
        self.q_dw = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=bias)
        self.kv_dw = nn.Conv2d(dim * 4, dim * 4, 3, padding=1, groups=dim * 4, bias=bias)

        self.norm_corr = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.proj_out  = nn.Conv2d(dim * 2, dim, 1, bias=bias)

    def forward(self, x, evt):
        """x, evt: (B, C, H, W)"""
        x   = self.norm_img(x)
        evt = self.norm_evt(evt)

        q  = self.q_dw(self.q(x))          # (B, 2C, H, W)
        kv = self.kv_dw(self.kv(evt))
        k, v = kv.chunk(2, dim=1)          # each (B, 2C, H, W)

        p = self.patch_size
        q_p = rearrange(q, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p)
        k_p = rearrange(k, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=p, p2=p)

        corr = torch.fft.rfft2(q_p.float()) * torch.fft.rfft2(k_p.float())
        corr = torch.fft.irfft2(corr, s=(p, p))
        corr = rearrange(corr, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=p, p2=p)
        corr = self.norm_corr(corr)

        return self.proj_out(v * corr)      # (B, C, H, W)


# ---------------------------------------------------------------------------
# CrossModalFuse  (fixed Event_Image_Fuse)
# ---------------------------------------------------------------------------

class CrossModalFuse(nn.Module):
    """Fuses image + event via CrossModalFFTAttn followed by DFFN (FFT FFN)."""

    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super().__init__()
        self.attn = CrossModalFFTAttn(dim, bias)
        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')
        self.ffn  = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, evt):
        x = x + self.attn(x, evt)
        x = x + self.ffn(self.norm(x))
        return x


# ---------------------------------------------------------------------------
# MambaFFTBlock — core hybrid block
# ---------------------------------------------------------------------------

class MambaFFTBlock(nn.Module):
    """
    Hybrid residual block combining:
      1. NSS Mamba SSM (spatial sequence modeling with locality-preserving scanning)
      2. DFFN         (patch-level FFT frequency-domain feed-forward network)

    Input / output: (B, C, H, W).
    Requires losh_ids = (scan_ids, inverse_ids) pre-computed for the current scale.
    """

    def __init__(self, dim, ssm_ratio=2.0, d_state=16,
                 ffn_expansion_factor=2, bias=False, drop_path=0.0):
        super().__init__()
        # Branch 1: NSS Mamba SSM
        self.norm1      = nn.LayerNorm(dim)
        self.ssm        = NSS_SSM(d_model=dim, ssm_ratio=ssm_ratio, d_state=d_state)
        self.drop_path  = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.skip_scale = nn.Parameter(torch.ones(dim))

        # Branch 2: FFT FFN
        self.norm2       = LayerNorm(dim, LayerNorm_type='WithBias')
        self.ffn         = DFFN(dim, ffn_expansion_factor, bias)
        self.skip_scale2 = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x, losh_ids):
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape

        # SSM branch — convert to (B, H, W, C) for NSS_SSM
        x_hwc  = x.permute(0, 2, 3, 1).contiguous()
        ssm_out = self.ssm(self.norm1(x_hwc), losh_ids)          # (B, H, W, C)
        ssm_out = ssm_out.permute(0, 3, 1, 2).contiguous()       # (B, C, H, W)
        x = x * self.skip_scale.view(1, C, 1, 1) + self.drop_path(ssm_out)

        # FFT FFN branch
        x = x * self.skip_scale2 + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# EventFFTMamba — main model
# ---------------------------------------------------------------------------

class EventFFTMamba(nn.Module):
    """
    Event-guided image restoration using Mamba NSS scanning + FFT frequency domain.

    Follows the dual-stream U-Net of eventfft_cross but replaces image-stream
    TransformerBlocks with MambaFFTBlock (NSS SSM + DFFN).

    Args:
        evt_inp:               event input channels (default 6).
        img_inp:               image input channels (default 3).
        out_channels:          output channels.
        dim:                   base feature dimension.
        num_blocks:            [n_l1, n_l2, n_l3] blocks per encoder level.
        num_refinement_blocks: refinement blocks after decoder.
        ssm_ratio:             inner expansion ratio for NSS_SSM.
        d_state:               SSM state dimension at level 1 (doubles per level).
        ffn_expansion_factor:  DFFN expansion ratio.
        scan_len:              NSS stripe width (must divide H and W at each scale).
        dynamic_ids:           if True, recompute NSS IDs every forward pass.
        img_size:              training image size for pre-computing NSS IDs.
    """

    def __init__(
        self,
        evt_inp=6,
        img_inp=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 4, 8),
        num_refinement_blocks=4,
        ssm_ratio=2.0,
        d_state=4,
        ffn_expansion_factor=2,
        bias=False,
        scan_len=8,
        dynamic_ids=False,
        img_size=256,
    ):
        super().__init__()
        self.scan_len         = scan_len
        self.dynamic_ids      = dynamic_ids
        self.training_img_size = img_size

        if not dynamic_ids:
            self._generate_ids(img_size, img_size)

        # ---- embeddings ----
        self.evt_embed = OverlapPatchEmbed(evt_inp, dim)
        self.img_embed = OverlapPatchEmbed(img_inp, dim)

        # ---- event encoder (FFT FFN only) ----
        self.evt_enc_l1  = nn.Sequential(*[
            Event_Transformer_Block(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[0])])
        self.evt_down1_2 = Downsample(dim)

        self.evt_enc_l2  = nn.Sequential(*[
            Event_Transformer_Block(dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[1])])
        self.evt_down2_3 = Downsample(dim * 2)

        self.evt_enc_l3  = nn.Sequential(*[
            Event_Transformer_Block(dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[2])])

        # ---- image encoder (NSS Mamba + FFT FFN) ----
        self.enc_l1  = nn.ModuleList([
            MambaFFTBlock(dim=dim, ssm_ratio=ssm_ratio, d_state=d_state,
                          ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)

        self.enc_l2  = nn.ModuleList([
            MambaFFTBlock(dim=dim * 2, ssm_ratio=ssm_ratio, d_state=d_state * 2,
                          ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)

        self.enc_l3  = nn.ModuleList([
            MambaFFTBlock(dim=dim * 4, ssm_ratio=ssm_ratio, d_state=d_state * 4,
                          ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[2])])

        # ---- cross-modal fusion (freq-domain cross-attention + DFFN) ----
        self.fuse_en1 = CrossModalFuse(dim,     ffn_expansion_factor, bias)
        self.fuse_en2 = CrossModalFuse(dim * 2, ffn_expansion_factor, bias)
        self.fuse_en3 = CrossModalFuse(dim * 4, ffn_expansion_factor, bias)

        # ---- decoder ----
        self.dec_l3  = nn.ModuleList([
            MambaFFTBlock(dim=dim * 4, ssm_ratio=ssm_ratio, d_state=d_state * 4,
                          ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[2])])
        self.up3_2   = Upsample(dim * 4)
        self.fuse2   = Fuse(dim * 2)

        self.dec_l2  = nn.ModuleList([
            MambaFFTBlock(dim=dim * 2, ssm_ratio=ssm_ratio, d_state=d_state * 2,
                          ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[1])])
        self.up2_1   = Upsample(dim * 2)
        self.fuse1   = Fuse(dim)

        self.dec_l1  = nn.ModuleList([
            MambaFFTBlock(dim=dim, ssm_ratio=ssm_ratio, d_state=d_state,
                          ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[0])])

        # ---- refinement ----
        self.refinement = nn.ModuleList([
            MambaFFTBlock(dim=dim, ssm_ratio=ssm_ratio, d_state=d_state,
                          ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_refinement_blocks)])

        self.output = nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)

    # ------------------------------------------------------------------
    # NSS index management
    # ------------------------------------------------------------------

    def _make_ids(self, h, w):
        scan, inv = mair_ids_generate((1, 1, h, w), scan_len=self.scan_len)
        if torch.cuda.is_available():
            return scan.cuda(), inv.cuda()
        return scan, inv

    def _generate_ids(self, H, W):
        self.ids_l1 = self._make_ids(H,     W)
        self.ids_l2 = self._make_ids(H // 2, W // 2)
        self.ids_l3 = self._make_ids(H // 4, W // 4)

    def _get_ids(self, H, W):
        if self.dynamic_ids or not self.training:
            return (
                self._make_ids(H,     W),
                self._make_ids(H // 2, W // 2),
                self._make_ids(H // 4, W // 4),
            )
        if self.training and H != self.training_img_size:
            self._generate_ids(H, W)
            self.training_img_size = H
        return self.ids_l1, self.ids_l2, self.ids_l3

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, event):
        """
        Args:
            x:     (B, img_inp, H, W) — blurry / noisy image.
            event: (B, evt_inp, H, W) — event voxel grid.
        Returns:
            (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        ids_l1, ids_l2, ids_l3 = self._get_ids(H, W)

        # ---- event encoder ----
        evt1 = self.evt_enc_l1(self.evt_embed(event))          # (B, dim,   H,   W)
        evt2 = self.evt_enc_l2(self.evt_down1_2(evt1))         # (B, dim*2, H/2, W/2)
        evt3 = self.evt_enc_l3(self.evt_down2_3(evt2))         # (B, dim*4, H/4, W/4)

        # ---- image encoder + cross-modal fusion ----
        f1 = self.img_embed(x)
        for blk in self.enc_l1:
            f1 = blk(f1, ids_l1)
        f1 = self.fuse_en1(f1, evt1)                           # (B, dim,   H,   W)

        f2 = self.down1_2(f1)
        for blk in self.enc_l2:
            f2 = blk(f2, ids_l2)
        f2 = self.fuse_en2(f2, evt2)                           # (B, dim*2, H/2, W/2)

        f3 = self.down2_3(f2)
        for blk in self.enc_l3:
            f3 = blk(f3, ids_l3)
        f3 = self.fuse_en3(f3, evt3)                           # (B, dim*4, H/4, W/4)

        # ---- decoder ----
        d3 = f3
        for blk in self.dec_l3:
            d3 = blk(d3, ids_l3)

        d2 = self.fuse2(self.up3_2(d3), f2)                   # (B, dim*2, H/2, W/2)
        for blk in self.dec_l2:
            d2 = blk(d2, ids_l2)

        d1 = self.fuse1(self.up2_1(d2), f1)                   # (B, dim,   H,   W)
        for blk in self.dec_l1:
            d1 = blk(d1, ids_l1)

        for blk in self.refinement:
            d1 = blk(d1, ids_l1)

        return self.output(d1) + x
