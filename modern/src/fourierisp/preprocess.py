"""ZRR-compatible Bayer packing. Other sensors require explicit calibration."""
import numpy as np
import torch
from torch import Tensor


def pack_rggb(mosaic: Tensor) -> Tensor:
    """Legacy channel order: bottom-right, top-right, top-left, bottom-left."""
    if mosaic.ndim < 2 or any(s % 2 or s < 2 for s in mosaic.shape[-2:]):
        raise ValueError('Bayer mosaic requires positive even H/W')
    return torch.stack((mosaic[..., 1::2, 1::2], mosaic[..., ::2, 1::2],
                        mosaic[..., ::2, ::2], mosaic[..., 1::2, ::2]), dim=-3)


def normalize_mosaic(mosaic: Tensor, *, black: float = 63, white: float = 1020) -> Tensor:
    if not np.isfinite([black, white]).all() or white <= black:
        raise ValueError('white must be finite and greater than black')
    return (mosaic.float() - black).clamp_min(0) / (white - black)


def preprocess_rggb(mosaic: np.ndarray, *, black: float = 63,
                    white: float = 1020) -> tuple[Tensor, Tensor]:
    """Return unbatched packed RAW and bilinear demosaiced RGB, as in legacy ZRR."""
    if mosaic.ndim != 2:
        raise ValueError('Expected one single-channel, undemosaiced RGGB mosaic')
    try:
        from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
    except ImportError as exc:
        raise RuntimeError('Install fourierisp-modern[raw] for Bayer demosaicing') from exc
    normalized = normalize_mosaic(torch.from_numpy(mosaic.astype(np.float32)), black=black, white=white)
    packed = pack_rggb(normalized)
    rgb = demosaicing_CFA_Bayer_bilinear(normalized.numpy(), pattern='RGGB')
    return packed, torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32))
