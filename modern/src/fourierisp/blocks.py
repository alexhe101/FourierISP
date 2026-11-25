"""Reusable spatial and frequency blocks; legacy parameter names are retained locally."""
from typing import Literal

import torch
from torch import Tensor, nn

from .spectral import fft_input, precision_context, real_spectrum

SpectralMode = Literal['none', 'phase', 'amplitude']


def conv3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


def downsample(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)


class HalfInstanceResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_1 = conv3(channels, channels)
        self.conv_2 = conv3(channels, channels)
        self.norm = nn.InstanceNorm2d(channels // 2, affine=True)
        # The original HIN blocks use slope 0.2 independently of the U-Net slope.
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        normalized, untouched = self.activation(self.conv_1(x)).chunk(2, dim=1)
        mixed = torch.cat((self.norm(normalized), untouched), dim=1)
        return x + self.activation(self.conv_2(mixed))


def stem(in_channels: int, width: int, blocks: int = 2) -> nn.Sequential:
    return nn.Sequential(conv3(in_channels, width), *(HalfInstanceResidual(width) for _ in range(blocks)))


class FourierResidual(nn.Module):
    def __init__(self, in_channels: int, channels: int, slope: float,
                 *, mode: SpectralMode = 'none', reduce: bool = False, cross_stage: bool = False):
        super().__init__()
        self.mode = mode
        self.identity = nn.Conv2d(in_channels, channels, 1)
        self.conv_1 = conv3(in_channels, channels)
        self.conv_2 = conv3(channels, channels)
        self.activation = nn.LeakyReLU(slope)
        if mode != 'none':
            self.conv_fft_1 = nn.Conv2d(channels, channels, 1)
            self.conv_fft_2 = nn.Conv2d(channels, channels, 1)
        self.downsample = downsample(channels, channels) if reduce else None
        if cross_stage:
            self.csff_enc = conv3(channels, channels)
            self.csff_dec = conv3(channels, channels)
        self.cross_stage = cross_stage

    def spectral_residual(self, x: Tensor) -> Tensor:
        with precision_context(x):
            spectrum = torch.fft.fft2(fft_input(x), dim=(-2, -1))
            amplitude, phase = spectrum.abs(), spectrum.angle()
            if self.mode == 'phase':
                phase = self.conv_fft_2(self.activation(self.conv_fft_1(phase)))
            else:
                amplitude = self.conv_fft_2(self.activation(self.conv_fft_1(amplitude)))
            result = torch.fft.ifft2(amplitude * torch.exp(1j * phase), dim=(-2, -1)).real
        return result.to(x.dtype)

    def forward(self, x: Tensor, context: tuple[Tensor, Tensor] | None = None) -> Tensor:
        identity = self.identity(x)
        out = self.activation(self.conv_2(self.activation(self.conv_1(x)))) + identity
        if self.mode != 'none':
            out = out + self.spectral_residual(identity)
        if self.cross_stage:
            if context is None:
                raise ValueError('Cross-stage encoder and decoder features are required')
            out = out + self.csff_enc(context[0]) + self.csff_dec(context[1])
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, slope: float):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, channels, 2, stride=2)
        self.conv_block = FourierResidual(2 * channels, channels, slope)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return self.conv_block(torch.cat((self.up(x), skip), dim=1))


class SupervisedAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = conv3(channels, channels)
        self.conv2 = conv3(channels, 3)
        self.conv3 = conv3(3, channels)

    def forward(self, features: Tensor, image: Tensor) -> tuple[Tensor, Tensor]:
        prediction = self.conv2(features) + image
        return self.conv1(features) * self.conv3(prediction).sigmoid() + features, prediction


class ChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.process = nn.Sequential(conv3(channels, channels), nn.ReLU(), conv3(channels, channels))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(channels, channels // 4, 1), nn.ReLU(),
                                     nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        residual = self.process(x)
        return x + residual * self.conv_du(self.avg_pool(residual))


class Refine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_in = conv3(channels, channels)
        self.process = nn.Sequential(ChannelAttention(channels))
        self.conv_last = conv3(channels, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_last(self.process(self.conv_in(x)))


class SpatialFeatureTransform(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.convmul = conv3(channels, channels)
        self.convadd = conv3(channels, channels)
        self.convfuse = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        return self.convfuse(torch.cat((x, self.convmul(condition) * x + self.convadd(condition)), 1))


class FrequencyAdjustment(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        def mixer():
            return nn.Sequential(nn.Conv2d(channels, channels, 1), nn.LeakyReLU(0.1),
                                 nn.Conv2d(channels, channels, 1))
        self.processmag = mixer()
        self.processpha = mixer()
        self.sft = SpatialFeatureTransform(channels)
        self.cat = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x: Tensor, amplitude: Tensor, phase: Tensor) -> Tensor:
        mag = self.sft(self.processmag(x.abs()), amplitude)
        pha = self.cat(torch.cat((phase, self.processpha(x.angle())), 1))
        # Unlike torch.polar, this also preserves the original signed magnitude behavior.
        return torch.complex(mag * pha.cos(), mag * pha.sin())


class ColorAdaptationBlock(nn.Module):
    def __init__(self, channels: int, *, propagate: bool):
        super().__init__()
        self.spatial_process = HalfInstanceResidual(channels)
        self.frequency_process = FrequencyAdjustment(channels)
        self.cat = nn.Conv2d(2 * channels, channels, 1)
        self.transamp = nn.Conv2d(channels, channels, 1)
        self.transpha = nn.Conv2d(channels, channels, 1)
        self.down = downsample(channels, 2 * channels) if propagate else None

    def forward(self, x: Tensor, color: Tensor) -> tuple[Tensor, Tensor | None]:
        spatial = self.spatial_process(x)
        with precision_context(x):
            spectrum = real_spectrum(x)
            condition = real_spectrum(color)
            adjusted = self.frequency_process(spectrum, self.transamp(condition.abs()),
                                              self.transpha(condition.angle()))
            frequency = torch.fft.irfft2(adjusted, s=x.shape[-2:], norm='backward')
        out = self.cat(torch.cat((spatial, frequency.to(spatial.dtype)), 1)) + x
        return out, self.down(color) if self.down is not None else None
