import pytest
import torch

from fourierisp import FourierISP
from fourierisp.checkpoint import PREFIXES, load_weights, migrate_legacy_state_dict, unwrap_checkpoint


def synthetic_legacy(model):
    result = {}
    for new, value in model.state_dict().items():
        old = next(src + new[len(dst):] for src, dst in PREFIXES.items() if new.startswith(dst))
        result[old] = value.clone()
    # Examples of legacy-only dead tensors; true parity is tested separately.
    result['p_net.pre_re.0.weight'] = torch.zeros(1)
    result['cat12.weight'] = torch.zeros(1)
    result['cas.adj3.down.weight'] = torch.zeros(1)
    result['cas.adj1.frequency_spatial.weight'] = torch.zeros(1)
    result['conv_01.1.identity.weight'] = torch.zeros(1)
    result['conv_01.1.conv_3.weight'] = torch.zeros(1)
    result['down_path_2.0.conv_fft_1.weight'] = torch.zeros(1)
    return result


def test_migration_and_round_trip(tmp_path):
    model = FourierISP(width=4)
    original = synthetic_legacy(model)
    converted, ignored = migrate_legacy_state_dict(model, original)
    assert len(ignored) == 7
    assert set(converted) == set(model.state_dict())
    path = tmp_path / 'legacy.pth'
    torch.save({'params_ema': {'module.'+k: v for k, v in original.items()}}, path)
    restored = FourierISP(width=4)
    assert len(load_weights(restored, path)) == 7
    for k, v in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[k], v)
    modern = tmp_path / 'modern.pth'
    torch.save({'params': restored.state_dict()}, modern)
    assert load_weights(model, modern) == []


def test_migration_rejects_unknown_or_missing():
    model = FourierISP(width=4)
    legacy = synthetic_legacy(model)
    legacy['unexpected.weight'] = torch.zeros(1)
    with pytest.raises(ValueError, match='unexpected.weight'):
        migrate_legacy_state_dict(model, legacy)
    del legacy['unexpected.weight']
    del legacy['conv_01.0.weight']
    with pytest.raises(ValueError, match='amplitude_stage.stem.0.weight'):
        migrate_legacy_state_dict(model, legacy)


def test_ema_priority():
    wrapper = {'params': {'x': torch.tensor(1)}, 'params_ema': {'x': torch.tensor(2)}}
    assert unwrap_checkpoint(wrapper)['x'].item() == 2
    assert unwrap_checkpoint(wrapper, prefer_ema=False)['x'].item() == 1
