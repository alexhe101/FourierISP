"""Execute the untouched upstream architecture without importing BasicSR's other dependencies."""
import os
import runpy
import sys
import types
from pathlib import Path

import pytest
import torch

from fourierisp import FourierISP
from fourierisp.checkpoint import migrate_legacy_state_dict


@pytest.mark.parametrize('width,size', [(4, 48), (24, 64)])
def test_legacy_forward_and_input_gradients(monkeypatch, width, size):
    legacy_path = Path(os.environ.get('FOURIERISP_LEGACY_ARCH',
                       Path(__file__).resolve().parents[2] / 'basicsr/archs/FourierISP_arch.py'))
    if not legacy_path.exists():
        pytest.skip('Original architecture not present; run from the merged repository')
    registry = types.ModuleType('basicsr.utils.registry')
    registry.ARCH_REGISTRY = types.SimpleNamespace(register=lambda: lambda obj: obj)
    monkeypatch.setitem(sys.modules, 'basicsr', types.ModuleType('basicsr'))
    monkeypatch.setitem(sys.modules, 'basicsr.utils', types.ModuleType('basicsr.utils'))
    monkeypatch.setitem(sys.modules, 'basicsr.utils.registry', registry)
    original = runpy.run_path(str(legacy_path))['FourierISP'](wf=width, depth=4).eval()
    model = FourierISP(width=width, depth=4).eval()
    mapped, ignored = migrate_legacy_state_dict(model, original.state_dict())
    assert ignored
    model.load_state_dict(mapped, strict=True)
    raw = torch.rand(1, 4, size//2, size//2, requires_grad=True)
    demo = torch.rand(1, 3, size, size, requires_grad=True)
    a, b = original(raw, demo), model(raw, demo)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, atol=3e-5, rtol=1e-5)
    loss = lambda result: result[0].abs().mean() + result[1].mean() + result[2].square().mean()
    grads_a = torch.autograd.grad(loss(a), (raw, demo))
    grads_b = torch.autograd.grad(loss(b), (raw, demo))
    for x, y in zip(grads_a, grads_b, strict=True):
        torch.testing.assert_close(x, y, atol=3e-5, rtol=1e-4)
