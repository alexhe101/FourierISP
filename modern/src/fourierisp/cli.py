"""Inference, checkpoint migration and synthetic latency benchmark."""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .checkpoint import load_weights
from .model import FourierISP
from .preprocess import preprocess_rggb


def build_model(args: argparse.Namespace) -> FourierISP:
    model = FourierISP(width=args.width, depth=args.depth, relu_slope=args.relu_slope)
    if args.checkpoint:
        ignored = load_weights(model, args.checkpoint, prefer_ema=not args.no_ema)
        if ignored:
            print(f'Migrated checkpoint; ignored {len(ignored)} known unused legacy tensors')
    return model


def device_for(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def infer(args: argparse.Namespace) -> None:
    device = device_for(args.device)
    model = build_model(args).to(device).eval()
    if args.input.suffix.lower() == '.npz':
        with np.load(args.input, allow_pickle=False) as blob:
            raw = torch.from_numpy(np.asarray(blob['raw'], dtype=np.float32))
            demo = torch.from_numpy(np.asarray(blob['demo'], dtype=np.float32))
    else:
        with Image.open(args.input) as image:
            raw, demo = preprocess_rggb(np.asarray(image), black=args.black, white=args.white)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
        output = model(raw.unsqueeze(0).to(device), demo.unsqueeze(0).to(device)).srgb
    rgb = (output[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output)


def benchmark(args: argparse.Namespace) -> None:
    device = device_for(args.device)
    model = build_model(args).to(device).eval()
    if args.compile:
        model = torch.compile(model)
    raw = torch.rand(1, 4, args.height // 2, args.image_width // 2, device=device)
    demo = torch.rand(1, 3, args.height, args.image_width, device=device)
    if args.height < 2 or args.image_width < 2 or args.height % 2 or args.image_width % 2:
        raise ValueError('Benchmark RGB dimensions must be positive and even')
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError('warmup and repeats must be >= 1')
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    durations = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
        for _ in range(args.warmup):
            model(raw, demo)
        synchronize()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.repeats):
            synchronize()
            start = time.perf_counter()
            model(raw, demo)
            synchronize()
            durations.append((time.perf_counter() - start) * 1000)
    print(f'device={device}, median={statistics.median(durations):.2f} ms, repeats={len(durations)}')
    if device.type == 'cuda':
        print(f'peak_allocated={torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB')


def convert(args: argparse.Namespace) -> None:
    model = build_model(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'params': model.state_dict(), 'width': args.width, 'depth': args.depth,
                'relu_slope': args.relu_slope}, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    for name, func in [('infer', infer), ('benchmark', benchmark), ('convert', convert)]:
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument('--width', type=int, default=24, help='Model base width')
        p.add_argument('--depth', type=int, default=4)
        p.add_argument('--relu-slope', type=float, default=0.2)
        p.add_argument('--checkpoint', type=Path, required=name != 'benchmark')
        p.add_argument('--no-ema', action='store_true', help='Prefer regular over EMA weights')
        if name in ('infer', 'benchmark'):
            p.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
            p.add_argument('--amp', action='store_true', help='BF16 autocast; spectral FP32')
        if name == 'infer':
            p.add_argument('--input', type=Path, required=True)
            p.add_argument('--output', type=Path, required=True)
            p.add_argument('--black', type=float, default=63)
            p.add_argument('--white', type=float, default=1020)
        elif name == 'convert':
            p.add_argument('--output', type=Path, required=True)
        else:
            p.add_argument('--height', type=int, default=448)
            p.add_argument('--image-width', type=int, default=448)
            p.add_argument('--warmup', type=int, default=3)
            p.add_argument('--repeats', type=int, default=10)
            p.add_argument('--compile', action='store_true', help='Experimental, benchmark before enabling')
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
