"""Spectral operations. Keep FFT arithmetic in FP32 (FP64 for double inputs).

Full FFT + real projection is intentionally retained for FPRB/FARB: learned
phase transforms need not preserve Hermitian symmetry.
"""
import torch
from torch import Tensor


def precision_context(x: Tensor):
    return torch.autocast(device_type=x.device.type, enabled=False)


def fft_input(x: Tensor) -> Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def real_spectrum(x: Tensor) -> Tensor:
    with precision_context(x):
        return torch.fft.rfft2(fft_input(x), dim=(-2, -1))


def phase_substitution(image: Tensor, phase: Tensor) -> Tensor:
    with precision_context(image):
        amplitude = real_spectrum(image).abs()
        spectrum = amplitude * torch.exp(1j * phase)
        return torch.fft.irfft2(spectrum, s=image.shape[-2:], dim=(-2, -1)).to(image.dtype)
