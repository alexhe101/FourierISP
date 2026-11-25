"""Paper-faithful model topology, separated from the training framework."""
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import (ColorAdaptationBlock, FourierResidual, HalfInstanceResidual, Refine,
                     SpectralMode, SupervisedAttention, UpBlock, conv3, stem)
from .spectral import phase_substitution, real_spectrum


class ISPOutput(NamedTuple):
    phase: Tensor
    amplitude: Tensor
    srgb: Tensor


class Pyramid(NamedTuple):
    bottleneck: Tensor
    skips: list[Tensor]


class UNet(nn.Module):
    def __init__(self, width: int, depth: int, slope: float, *,
                 mode: SpectralMode = 'none', cross_stage: bool = False):
        super().__init__()
        channels = [width * 2**i for i in range(depth)]
        self.encoder = nn.ModuleList(
            FourierResidual(channels[max(0, i-1)], c, slope, mode=mode,
                            reduce=i < depth-1, cross_stage=cross_stage and i < depth-1)
            for i, c in enumerate(channels))
        self.decoder = nn.ModuleList(UpBlock(channels[i+1], channels[i], slope)
                                     for i in reversed(range(depth-1)))
        self.skip = nn.ModuleList(HalfInstanceResidual(channels[i])
                                  for i in reversed(range(depth-1)))

    def encode(self, x: Tensor, context: list[tuple[Tensor, Tensor]] | None = None) -> Pyramid:
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x, context[i] if context is not None and i < len(context) else None)
            if block.downsample is not None:
                skips.append(x)
                x = block.downsample(x)
        return Pyramid(x, skips)

    def decode(self, pyramid: Pyramid) -> tuple[Tensor, list[Tensor]]:
        x = pyramid.bottleneck
        decoded = []
        for up, refine, skip in zip(self.decoder, self.skip, reversed(pyramid.skips), strict=True):
            x = up(x, refine(skip))
            decoded.append(x)
        return x, decoded


class PhaseResult(NamedTuple):
    phase: Tensor
    features: Tensor
    substituted: Tensor
    skips: list[Tensor]
    decoded: list[Tensor]


class PhaseEnhanceSubnet(nn.Module):
    def __init__(self, width: int, depth: int, slope: float):
        super().__init__()
        self.stem = stem(4, width)
        # Legacy PhaseNet always used 0.2, even when the top-level slope differed.
        self.unet = UNet(width, depth, 0.2, mode='phase')
        self.demosaic = nn.Sequential(conv3(width, 4 * width), nn.PixelShuffle(2))
        self.attention = SupervisedAttention(width)

    def forward(self, raw: Tensor, demo: Tensor) -> PhaseResult:
        pyramid = self.unet.encode(self.stem(raw))
        x, decoded = self.unet.decode(pyramid)
        features, image = self.attention(self.demosaic(x), demo)
        phase = real_spectrum(image).angle()
        return PhaseResult(phase, features, phase_substitution(demo, phase), pyramid.skips, decoded)


class AmplitudeRefineSubnet(nn.Module):
    def __init__(self, width: int, depth: int, slope: float):
        super().__init__()
        self.stem = stem(3, width)
        self.unet = UNet(width, depth, slope, mode='amplitude')
        self.attention = SupervisedAttention(width)

    def forward(self, demo: Tensor) -> tuple[Tensor, Tensor]:
        x, _ = self.unet.decode(self.unet.encode(self.stem(demo)))
        features, image = self.attention(x, demo)
        return real_spectrum(image).abs(), features


class ColorAdaptationSubnet(nn.Module):
    def __init__(self, width: int, depth: int, slope: float):
        super().__init__()
        self.stem = stem(3, width)
        self.merge = nn.Conv2d(3 * width, width, 1)
        self.unet = UNet(width, depth, slope, cross_stage=True)
        self.adapters = nn.ModuleList(ColorAdaptationBlock(width * 2**i, propagate=i < depth-2)
                                      for i in range(depth-1))
        self.head = Refine(width)
        self.refine_stem = stem(3, width, blocks=1)
        self.refine_head = Refine(width)

    def forward(self, phase: PhaseResult, color: Tensor, demo: Tensor) -> Tensor:
        x = self.merge(torch.cat((self.stem(phase.substituted), phase.features, color), 1))
        context = [(F.interpolate(enc, scale_factor=2, mode='bilinear', align_corners=False),
                    F.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=False))
                   for enc, dec in zip(phase.skips, reversed(phase.decoded), strict=True)]
        pyramid = self.unet.encode(x, context)
        adjusted: list[Tensor] = []
        condition: Tensor | None = color
        for block, skip in zip(self.adapters, pyramid.skips, strict=True):
            if condition is None:
                raise RuntimeError("Color pyramid ended before all adaptation blocks")
            skip, condition = block(skip, condition)
            adjusted.append(skip)
        x, _ = self.unet.decode(Pyramid(pyramid.bottleneck, adjusted))
        image = self.head(x) + demo
        return self.refine_head(self.refine_stem(image)) + image


class FourierISP(nn.Module):
    """Inputs: normalized packed Bayer [B,4,h,w] and demosaiced [B,3,2h,2w].

    Keep model parameters FP32 for mixed precision. Outputs are not clamped.
    For depth=4 (paper config), packed H/W must be divisible by 8.
    """
    def __init__(self, *, width: int = 24, depth: int = 4, relu_slope: float = 0.2):
        super().__init__()
        if width < 4 or width % 2:
            raise ValueError('width must be even and >= 4')
        if depth < 2:
            raise ValueError('depth must be >= 2')
        self.width, self.depth, self.relu_slope = width, depth, relu_slope
        self.phase_stage = PhaseEnhanceSubnet(width, depth, relu_slope)
        self.amplitude_stage = AmplitudeRefineSubnet(width, depth, relu_slope)
        self.color_stage = ColorAdaptationSubnet(width, depth, relu_slope)

    def forward(self, raw: Tensor, demo: Tensor) -> ISPOutput:
        if raw.ndim != 4 or demo.ndim != 4 or raw.shape[1] != 4 or demo.shape[1] != 3:
            raise ValueError('Expected NCHW tensors with 4 RAW and 3 RGB channels')
        if raw.shape[0] != demo.shape[0] or tuple(demo.shape[-2:]) != tuple(2*s for s in raw.shape[-2:]):
            raise ValueError('Demosaiced spatial size must be twice packed RAW size')
        multiple = 2**(self.depth-1)
        if any(s < multiple or s % multiple for s in raw.shape[-2:]):
            raise ValueError(f'Packed RAW H/W must be positive multiples of {multiple}')
        if not raw.is_floating_point() or not demo.is_floating_point():
            raise TypeError('Expected preprocessed floating-point inputs')
        if raw.device != demo.device or raw.dtype != demo.dtype:
            raise ValueError('RAW and demosaiced inputs must share dtype and device')
        phase = self.phase_stage(raw, demo)
        amplitude, color = self.amplitude_stage(demo)
        return ISPOutput(phase.phase, amplitude, self.color_stage(phase, color, demo))
