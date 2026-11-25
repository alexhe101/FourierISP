"""Strict migration of active legacy FourierISP weights (depth=4)."""
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from .model import FourierISP

# Prefix rules map the active legacy architecture into the modular namespace.
PREFIXES = {
    'p_net.conv_01.': 'phase_stage.stem.',
    'p_net.down_path_1.': 'phase_stage.unet.encoder.',
    'p_net.up_path_1.': 'phase_stage.unet.decoder.',
    'p_net.skip_conv_1.': 'phase_stage.unet.skip.',
    'p_net.demosaic.': 'phase_stage.demosaic.',
    'p_net.sam12.': 'phase_stage.attention.',
    'conv_01.': 'amplitude_stage.stem.',
    'down_path_1.': 'amplitude_stage.unet.encoder.',
    'up_path_1.': 'amplitude_stage.unet.decoder.',
    'skip_conv_1.': 'amplitude_stage.unet.skip.',
    'sam12.': 'amplitude_stage.attention.',
    'conv_02.': 'color_stage.stem.',
    'cat123.': 'color_stage.merge.',
    'down_path_2.': 'color_stage.unet.encoder.',
    'up_path_2.': 'color_stage.unet.decoder.',
    'skip_conv_2.': 'color_stage.unet.skip.',
    'cas.adj1.': 'color_stage.adapters.0.',
    'cas.adj2.': 'color_stage.adapters.1.',
    'cas.adj3.': 'color_stage.adapters.2.',
    'last.': 'color_stage.head.',
    'pre_re.': 'color_stage.refine_stem.',
    'last2.': 'color_stage.refine_head.',
}


def unwrap_checkpoint(blob: Mapping, *, prefer_ema: bool = True) -> dict[str, Tensor]:
    order = ('params_ema', 'params', 'state_dict', 'model') if prefer_ema else (
        'params', 'state_dict', 'model', 'params_ema')
    for key in order:
        if key in blob and isinstance(blob[key], Mapping):
            blob = blob[key]
            break
    if not blob or not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in blob.items()):
        raise ValueError('Expected a tensor-only state dict or a recognized checkpoint wrapper')
    out = {}
    for key, value in blob.items():
        key = key.removeprefix('module.').removeprefix('_orig_mod.')
        if key in out:
            raise ValueError(f'Duplicate key after prefix removal: {key}')
        out[key] = value
    return out


def _known_unused(old: str, new: str, model: FourierISP) -> bool:
    if old.startswith(('p_net.pre_re.', 'cat12.', 'cas.adj3.down.')):
        return True
    if old.startswith('cas.') and any(f'.{k}.' in old for k in ('frequency_spatial', 'spatial_frequency')):
        return True
    # HIN's legacy identity projection and conv_3 are never used by forward.
    if any(f'.{k}.' in new for k in ('identity', 'conv_3')):
        prefix = new.rsplit('.', 1)[0]
        if prefix.rsplit('.', 1)[-1] in ('identity', 'conv_3'):
            parent = prefix.rsplit('.', 1)[0]
            try:
                from .blocks import HalfInstanceResidual
                if isinstance(model.get_submodule(parent), HalfInstanceResidual):
                    return True
            except AttributeError:
                pass
    if '.conv_fft_' in new:
        parent = new.rsplit('.', 2)[0]
        try:
            from .blocks import FourierResidual
            module = model.get_submodule(parent)
            return isinstance(module, FourierResidual) and module.mode == 'none'
        except AttributeError:
            pass
    return False


def migrate_legacy_state_dict(model: FourierISP, state: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], list[str]]:
    if model.depth != 4:
        raise ValueError('Legacy checkpoint migration requires depth=4')
    expected = model.state_dict()
    converted, ignored, unknown = {}, [], []
    for old, value in state.items():
        new = next((dst + old[len(src):] for src, dst in PREFIXES.items() if old.startswith(src)), '')
        if new in expected:
            if new in converted or value.shape != expected[new].shape:
                raise ValueError(f'Duplicate or mismatched shape: {old} -> {new}')
            converted[new] = value
        elif _known_unused(old, new, model):
            ignored.append(old)
        else:
            unknown.append(old)
    missing = sorted(expected.keys() - converted.keys())
    if missing or unknown:
        raise ValueError(f'Legacy checkpoint mismatch; missing={missing}; unknown={unknown}')
    return converted, sorted(ignored)


def load_weights(model: FourierISP, path: str | Path, *, prefer_ema: bool = True) -> list[str]:
    blob = torch.load(path, map_location='cpu', weights_only=True)
    state = unwrap_checkpoint(blob, prefer_ema=prefer_ema)
    ignored = []
    if set(state) != set(model.state_dict()):
        state, ignored = migrate_legacy_state_dict(model, state)
    model.load_state_dict(state, strict=True)
    return ignored
