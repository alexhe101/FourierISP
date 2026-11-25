# FourierISP Modern

An additive, standalone PyTorch refactor of [FourierISP (AAAI 2024)](https://arxiv.org/abs/2401.02161). The original BasicSR architecture, datasets, and experiments remain untouched. The default network (`width=24`, `depth=4`, `relu_slope=0.2`) is intended to reproduce the active legacy computation, not to introduce a new ISP method.

## Install

From the original repository after applying the modern branch patch:

```sh
python -m venv .venv
source .venv/bin/activate
# Install the appropriate PyTorch build for your machine first if necessary.
python -m pip install -e './modern[dev,raw]'
python -m pytest modern/tests -q -rs
```

Python 3.10+ and PyTorch 2.4+. The standalone model does not require BasicSR, torchvision, OpenCV or CUDA. The optional `raw` extra is needed for bilinear Bayer demosaicing from a single-channel mosaic. For legacy training, install the upstream BasicSR training dependencies separately.

## Architecture

| Paper stage | Implementation | Responsibility |
| --- | --- | --- |
| PES | `PhaseEnhanceSubnet` | Packed Bayer U-Net, FPRB, pixel shuffle, phase supervision and substitution |
| ARS | `AmplitudeRefineSubnet` | Demosaiced input U-Net, FARB, amplitude supervision and color features |
| CAS | `ColorAdaptationSubnet` | Cross-stage fusion, multiscale CABs, spatial/frequency refinement and residual RGB prediction |

The three stages share an encoder-decoder abstraction. Frequency operators are isolated in `spectral.py`. Full complex `fft2`/`ifft2` is preserved for FPRB/FARB, since learned phase transforms need not preserve Hermitian symmetry. The original real-spectrum operations remain real FFTs. Signed learned magnitudes are retained. FFT arithmetic and its learned frequency-domain convolutions run in FP32 under autocast (FP64 for double-precision models); keep model weights in FP32 for mixed precision. This avoids half-precision FFT shape restrictions and reduces overflow risk. It does not guarantee identical AMP outputs to FP32.

```python
import torch
from fourierisp import FourierISP
from fourierisp.checkpoint import load_weights

model = FourierISP(width=24, depth=4).eval()
# load_weights(model, 'your_checkpoint.pth')
raw = torch.rand(1, 4, 112, 112)
demo = torch.rand(1, 3, 224, 224)
with torch.inference_mode():
    result = model(raw, demo)
print(result.srgb.shape)       # [1, 3, 224, 224]
print(result.phase.shape)      # [1, 3, 224, 113]
print(result.amplitude.shape)  # [1, 3, 224, 113]
```

The packed Bayer channel order is `[B, G(top-right), R, G(bottom-left)]` (legacy ZRR convention). Both input tensors must be floating-point NCHW, share device/dtype, and have a 1:2 spatial relationship. Packed height and width must be multiples of `2**(depth-1)`; at the default depth, RGB dimensions must be multiples of 16. No implicit resizing or padding is performed, because padding changes global frequency statistics. The network output is not clipped; the image-writing CLI clips it to `[0, 1]`.

The `width` must be even and at least 4. All stages now honor `depth>=2`; non-default depths are extensions, not checkpoint-compatible with the published configuration. For non-default `relu_slope`, PES intentionally retains the original hard-coded slope 0.2 for compatibility.

## Weights and inference

The upstream README states that no official checkpoint is publicly available. Use your own trained legacy or modern checkpoint. A randomly initialized model is only suitable for smoke tests.

```sh
# Converts legacy weights to the new namespace; defaults to EMA weights if present.
fourierisp convert --checkpoint legacy.pth --output modern.pth

# Run on a single, already-preprocessed NPZ (unbatched CHW arrays: raw, demo).
fourierisp infer --checkpoint modern.pth --input example.npz --output result.png

# Run on an undemosaiced single-channel RGGB mosaic (e.g. 16-bit PNG/TIFF).
# The following calibration is specific to ZRR, not a generic camera RAW pipeline.
fourierisp infer --checkpoint modern.pth --input mosaic.png --output result.png \
  --black 63 --white 1020 --device cuda --amp
```

The converter supports a tensor-only state dict and the common `params`, `params_ema`, `state_dict`, `model` wrappers. The conversion is strict: only known unused legacy tensors are discarded; unknown keys, shape mismatches, and missing active tensors fail. It does not migrate optimizer state, scheduler state or training RNG state. Pass `--no-ema` to convert the regular weights instead of EMA. Match the width and depth used to train the checkpoint. The CLI is not a DNG/CR2/NEF decoder; sensor calibration and CFA pattern must match the model's training data.

## Training with the original objective

The optional `FourierISPModern` BasicSR architecture adapter preserves the original positional/list output contract. The new configuration uses the original ZRR dataset, `FcolorDualModel` loss stack and training hyperparameters, while switching to the new architecture. Set your data directories first.

```sh
python basicsr/train.py -opt options/train/FourierISP/train_FourierISPModern.yml
```

Use a converted modern checkpoint when initializing the BasicSR adapter from legacy weights. If you need to preserve the original `params_ema` and `params` simultaneously for further training, convert each separately and construct a checkpoint containing both modern key namespaces. This refactor does not provide a native trainer replacement; preserving the research objective and validation pipeline is deliberate. An end-to-end training run is still required to validate BasicSR dependency compatibility and paper metrics.

## Benchmark

```sh
fourierisp benchmark --device cuda --height 448 --image-width 448 --warmup 5 --repeats 20
fourierisp benchmark --device cuda --height 448 --image-width 448 --warmup 5 --repeats 20 --amp
# Optional experimental compilation (measure before deciding to use it):
fourierisp benchmark --device cuda --height 448 --image-width 448 --compile
```

Benchmark reports synchronized median model latency and, on CUDA, peak allocated memory. It excludes file I/O and preprocessing. `--compile` may encounter complex FFT graph/backend limitations; it is not enabled by default. Parameter removal is not a claim of speedup or improved PSNR/SSIM.

## Tests and validation status

`pytest` covers shape validation, configurable depth, all active parameter gradients, BF16 CPU autocast, FFT roundtrip including odd sizes, Bayer normalization/channel order, checkpoint conversion, CLI inference and legacy parity. The parity test loads the *untouched upstream* architecture from `basicsr/archs/FourierISP_arch.py`, transfers its active weights, then compares all three outputs and input gradients. When testing the overlay alone, parity tests skip unless `FOURIERISP_LEGACY_ARCH` points to a reference file. CUDA tests skip without a GPU. CI checks the upstream file exists and runs the complete CPU suite after the overlay is applied to the original repository.

Local development validation used a manually transcribed reference from the public upstream code because the Git remote was unavailable. It verified the migration and numerical outputs on synthetic tensors, including the paper's default width. This is not a substitute for CI against the actual upstream file, actual trained checkpoint, full ZRR metrics, GPU latency, or a training run. No quality or performance improvement over the paper is claimed.

## References

- [Original repository](https://github.com/alexhe101/FourierISP)
- [Enhancing RAW-to-sRGB with Decoupled Style Structure in Fourier Domain (AAAI 2024)](https://arxiv.org/abs/2401.02161)
