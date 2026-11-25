import sys

import numpy as np
import torch
from PIL import Image

from fourierisp import FourierISP
from fourierisp.cli import main


def test_convert_and_infer_npz(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'input.pth'
    converted = tmp_path / 'converted.pth'
    sample = tmp_path / 'input.npz'
    output = tmp_path / 'output.png'
    torch.save({'params': FourierISP(width=4, depth=2).state_dict()}, checkpoint)
    raw = np.random.default_rng(42).random((4, 8, 12), dtype=np.float32)
    demo = np.random.default_rng(43).random((3, 16, 24), dtype=np.float32)
    np.savez(sample, raw=raw, demo=demo)
    monkeypatch.setattr(sys, 'argv', ['fourierisp', 'convert', '--width', '4', '--depth', '2',
                                     '--checkpoint', str(checkpoint), '--output', str(converted)])
    main()
    assert converted.exists()
    monkeypatch.setattr(sys, 'argv', ['fourierisp', 'infer', '--width', '4', '--depth', '2',
                                     '--checkpoint', str(converted), '--input', str(sample),
                                     '--output', str(output), '--device', 'cpu'])
    main()
    with Image.open(output) as result:
        assert result.mode == 'RGB'
        assert result.size == (24, 16)
