"""NSS (Nested S-shaped Scanning) index generation utilities.

Migrated from reference_1/realDenoising/basicsr/models/archs/shift_scanf_util.py
(MaIR, CVPR 2025). Removed hardcoded paths and test code.
"""
import torch
from einops import rearrange


def _create_idx(b, c, h, w):
    return torch.arange(b * c * h * w).reshape(b, c, h, w)


def _sscan(inp, scan_len, shift_len=0):
    """Apply Nested S-shaped scan reordering in-place on inp (B, C, H, W)."""
    B, C, H, W = inp.shape

    # Flip alternating stripe columns to create S-shaped vertical paths
    if shift_len == 0:
        for i in range(1, (W // scan_len) + 1, 2):
            inp[:, :, :, i * scan_len:(i + 1) * scan_len] = \
                inp[:, :, :, i * scan_len:(i + 1) * scan_len].flip(dims=[-2])
    else:
        for i in range(0, ((W - shift_len) // scan_len) + 1, 2):
            s = shift_len + i * scan_len
            e = shift_len + (i + 1) * scan_len
            inp[:, :, :, s:e] = inp[:, :, :, s:e].flip(dims=[-2])

    # Flip alternating sub-rows within each scan block
    if shift_len == 0:
        for hi in range(H // 2):
            for wi in range(W // scan_len):
                inp[:, :, 2 * hi + 1, wi * scan_len:(wi + 1) * scan_len] = \
                    inp[:, :, 2 * hi + 1, wi * scan_len:(wi + 1) * scan_len].flip(dims=[-1])
    else:
        for hi in range(H // 2):
            inp[:, :, 2 * hi + 1, :shift_len] = \
                inp[:, :, 2 * hi + 1, :shift_len].flip(dims=[-1])
            for wi in range((W - shift_len) // scan_len):
                s = shift_len + wi * scan_len
                e = shift_len + (wi + 1) * scan_len
                inp[:, :, 2 * hi + 1, s:e] = inp[:, :, 2 * hi + 1, s:e].flip(dims=[-1])

    # Handle right-boundary remainder
    remainder = (W - shift_len) % scan_len
    if remainder:
        tail = inp[:, :, :, -remainder:]
        tail[:, :, 1::2, :] = tail[:, :, 1::2, :].flip(dims=[-1])
        inp_last = tail.reshape(B, C, -1)
        inp_rest = inp[:, :, :, :-remainder]
    else:
        inp_last = None
        inp_rest = inp

    # Flatten windows into sequence
    if shift_len == 0:
        inp_window = rearrange(inp_rest, "b c h (d2 w) -> (b c d2) h w", w=scan_len)
        inp_flatten = inp_window.reshape(B, C, -1)
    else:
        inp_first = inp_rest[:, :, :, :shift_len].reshape(B, C, -1)
        inp_middle = inp_rest[:, :, :, shift_len:]
        inp_window = rearrange(inp_middle, "b c h (d2 w) -> (b c d2) h w", w=scan_len)
        inp_flatten = torch.cat((inp_first, inp_window.reshape(B, C, -1)), dim=-1)

    if inp_last is not None:
        inp_flatten = torch.cat((inp_flatten, inp_last), dim=-1)

    return inp_flatten


def _sscan_4d(inp, scan_len, shift_len=0):
    """Build 4-direction NSS sequences (original + reverse + transposed variants)."""
    B, C, H, W = inp.shape
    L = H * W

    inp_reverse = torch.flip(inp, dims=[-1, -2])
    inp_cat = torch.cat((inp, inp_reverse), dim=1)
    inp_cat_t = inp_cat.transpose(-1, -2).contiguous()

    line1 = _sscan(inp_cat, scan_len, shift_len)
    line2 = _sscan(inp_cat_t, scan_len, shift_len)

    xs = torch.stack(
        [line1.reshape(B, 2, -1, L), line2.reshape(B, 2, -1, L)], dim=1
    ).reshape(B, 4, -1, L)
    return xs


def mair_ids_generate(inp_shape, scan_len=4, K=4):
    """
    Pre-compute NSS scan and inverse permutation indices.

    Args:
        inp_shape: (B, C, H, W) — only H, W are used.
        scan_len:  stripe width for NSS scanning.
        K:         number of scan directions (fixed 4).

    Returns:
        xs_scan_ids:    (4, 1, L) — scan permutation indices.
        xs_inverse_ids: (4, 1, L) — argsort inverse indices.
    """
    _, _, h, w = inp_shape
    inp_idx = _create_idx(1, 1, h, w)
    xs_scan_ids = _sscan_4d(inp_idx, scan_len)[0]        # (4, 1, L)
    xs_inverse_ids = torch.argsort(xs_scan_ids, dim=-1)  # (4, 1, L)
    return xs_scan_ids, xs_inverse_ids


def mair_ids_scan(inp, xs_scan_ids, K=4):
    """
    Reorder feature map tokens according to 4-direction NSS scan indices.

    Args:
        inp:          (B, C, H, W)
        xs_scan_ids:  (4, 1, L) or (4, L)

    Returns:
        (B, 4, C*L)  — scanned sequences for all 4 directions.
    """
    B, C, H, W = inp.shape
    L = H * W
    xs_scan_ids = xs_scan_ids.reshape(K, L)
    flat = inp.reshape(B, 1, C, -1)
    ys = [torch.index_select(flat, -1, xs_scan_ids[k]) for k in range(K)]
    return torch.cat(ys, dim=1).reshape(B, 4, -1)   # (B, 4, C*L)


def mair_ids_inverse(inp, xs_inverse_ids, shape=None):
    """
    Inverse-permute scanned SSM outputs back to spatial layout.

    Args:
        inp:             (B, K, C, L) — selective_scan output reshaped.
        xs_inverse_ids:  (4, 1, L) or (4, L)
        shape:           optional (B, -1, H, W) to reshape spatial dims.

    Returns:
        Concatenated inverse tensor (B, K*C, L) or (B, K*C, H, W).
    """
    B, K, _, L = inp.shape
    xs_inverse_ids = xs_inverse_ids.reshape(K, L)
    ys = []
    for k in range(K):
        row = torch.index_select(inp[:, k, :], -1, xs_inverse_ids[k])
        if shape is not None:
            row = row.reshape(shape[0], shape[1], shape[2], shape[3])
        else:
            row = row.reshape(B, -1, L)
        ys.append(row)
    return torch.cat(ys, dim=1)
