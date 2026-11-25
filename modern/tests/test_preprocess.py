import torch
import pytest

from fourierisp.preprocess import normalize_mosaic, pack_rggb


def test_zrr_calibration_and_pack():
    mosaic = torch.tensor([[63, 1020], [0, 1023]])
    x = pack_rggb(normalize_mosaic(mosaic))
    torch.testing.assert_close(x.flatten(), torch.tensor([960/957, 1., 0., 0.]))


def test_bad_mosaic_and_calibration():
    with pytest.raises(ValueError):
        pack_rggb(torch.ones(3, 2))
    with pytest.raises(ValueError):
        normalize_mosaic(torch.ones(2, 2), black=100, white=99)
