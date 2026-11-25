import pytest
import torch

from fourierisp import FourierISP, ISPOutput
from fourierisp.spectral import phase_substitution, real_spectrum


@pytest.mark.parametrize('depth,size', [(2, 6), (3, 12), (4, 24)])
def test_forward_and_gradients(depth, size):
    model = FourierISP(width=4, depth=depth)
    raw = torch.rand(1, 4, size, size + 2**(depth-1))
    demo = torch.rand(1, 3, raw.shape[-2]*2, raw.shape[-1]*2)
    output = model(raw, demo)
    assert isinstance(output, ISPOutput)
    assert output.srgb.shape == demo.shape
    assert output.phase.shape == output.amplitude.shape == (*demo.shape[:-1], demo.shape[-1]//2+1)
    assert all(torch.isfinite(x).all() for x in output)
    loss = output.phase.abs().mean() + output.amplitude.mean() + output.srgb.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_real_fft_roundtrip_with_odd_width():
    image = torch.rand(2, 3, 15, 17)
    torch.testing.assert_close(phase_substitution(image, real_spectrum(image).angle()), image)


def test_invalid_inputs():
    model = FourierISP(width=4, depth=4)
    with pytest.raises(ValueError, match='twice'):
        model(torch.rand(1, 4, 8, 8), torch.rand(1, 3, 16, 18))
    with pytest.raises(ValueError, match='multiples'):
        model(torch.rand(1, 4, 9, 8), torch.rand(1, 3, 18, 16))
    with pytest.raises(ValueError, match='width'):
        FourierISP(width=3)


def test_cpu_bfloat16_backward():
    model = FourierISP(width=4, depth=3)
    raw, demo = torch.rand(1, 4, 12, 20), torch.rand(1, 3, 24, 40)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = model(raw, demo)
        loss = out.phase.abs().mean() + out.amplitude.mean() + out.srgb.square().mean()
    assert out.phase.dtype == out.amplitude.dtype == torch.float32
    assert all(torch.isfinite(x).all() for x in out)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_mixed_precision():
    model = FourierISP(width=4).cuda()
    raw, demo = torch.rand(1, 4, 24, 32, device='cuda'), torch.rand(1, 3, 48, 64, device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        out = model(raw, demo)
    assert all(torch.isfinite(x).all() for x in out)
