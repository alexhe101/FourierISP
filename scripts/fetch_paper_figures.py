#!/usr/bin/env python3
"""Fetch original arXiv figure panels, never synthesize model outputs.

If the HTML images are unavailable, render the relevant PDF figure regions.
Only arXiv v1 is used for the fallback layout. Keep the original labels.
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import fitz
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "paper"
HTML = "https://arxiv.org/html/2401.02161v1"
PDF = "https://arxiv.org/pdf/2401.02161v1"
FIGURES = {1: 0, 5: 4, 6: 5, 7: 5, 8: 6}


def get(url: str) -> bytes:
    last = None
    for _ in range(2):
        try:
            r = requests.get(url, timeout=(15, 45), headers={"User-Agent": "FourierISP-project-figures/1.0 (research project archive)"})
            r.raise_for_status()
            if len(r.content) > 20_000_000:
                raise ValueError("Unexpectedly large response")
            return r.content
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(1)
    raise RuntimeError(f"Could not retrieve {url}") from last


def html_panel(n: int) -> Image.Image:
    raw = get(f"{HTML}/x{n}.png")
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        if im.width < 200 or im.height < 50:
            raise ValueError("Unexpected image dimensions")
        return im.convert("RGB")


def pdf_panel(doc: fitz.Document, n: int) -> Image.Image:
    page = doc[FIGURES[n]]
    caps = [r for r in page.search_for(f"Figure {n}:")]
    if len(caps) != 1:
        raise ValueError(f"Expected one caption for figure {n}, got {len(caps)}")
    cap = caps[0]
    w, h = page.rect.width, page.rect.height
    # Figures 6 and 7 span both columns. Bounds were read from the v1 PDF.
    # Use exact caption text to confirm the version and keep a margin for labels.
    if n == 6:
        clip = fitz.Rect(.086*w, .232*h, .913*w, min(.557*h, cap.y0-2))
    elif n == 7:
        clip = fitz.Rect(.086*w, .600*h, .913*w, min(.727*h, cap.y0-2))
    else:
        # Single-column figures 1, 5, 8. Locate raster objects in that
        # column immediately above the caption, then preserve the labels.
        left = .08*w if cap.x0 < w/2 else .51*w
        right = .49*w if cap.x0 < w/2 else .92*w
        candidates = []
        for info in page.get_image_info():
            box = fitz.Rect(info["bbox"])
            if box.y1 <= cap.y0+2 and box.y0 >= .04*h and box.width >= 15 and box.height >= 15:
                if left-5 <= (box.x0+box.x1)/2 <= right+5:
                    candidates.append(box)
        if not candidates:
            raise ValueError(f"No raster panels above figure {n}")
        # Start from the bottom-most row and include nearby image rows.
        candidates.sort(key=lambda b: b.y1, reverse=True)
        top = candidates[0].y0
        for box in candidates[1:]:
            if box.y1 >= top-32:
                top = min(top, box.y0)
        clip = fitz.Rect(left, max(0, top-10), right, cap.y0-2)
    if clip.is_empty or clip.height < 20:
        raise ValueError(f"Invalid figure crop: {n}")
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def main() -> None:
    images = {}
    origins = {}
    doc = None
    for n in FIGURES:
        try:
            images[n] = html_panel(n)
            origins[n] = f"{HTML}/x{n}.png"
        except (RuntimeError, ValueError, OSError) as e:
            print(f"Figure {n}: HTML unavailable ({e}); trying PDF.")
            if doc is None:
                data = get(PDF)
                if not data.startswith(b"%PDF"):
                    raise ValueError("Not a PDF")
                doc = fitz.open(stream=data, filetype="pdf")
                if len(doc) != 9 or "Enhancing RAW-to-sRGB" not in doc[0].get_text():
                    raise ValueError("Unexpected arXiv PDF version")
            images[n] = pdf_panel(doc, n)
            origins[n] = f"{PDF}#page={FIGURES[n]+1}"
        print(f"Figure {n}: {images[n].size} from {origins[n]}")
    OUT.mkdir(parents=True, exist_ok=True)
    for n, im in images.items():
        im.save(OUT / f"figure-{n}.png", optimize=True)
    (OUT / "SOURCES.md").write_text(
        "# Original published figures\n\n"
        "He et al., AAAI 2024, DOI: 10.1609/aaai.v38i3.27985. "
        "These are extracted panels, not newly generated results. "
        "Original rights remain with their holders.\n\n"
        + "".join(f"- Figure {n}: {origins[n]}\n" for n in FIGURES), encoding="utf-8")
    # README uses remote arXiv images until local extraction succeeds.
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for n in FIGURES:
        text = text.replace(f"{HTML}/x{n}.png", f"docs/assets/paper/figure-{n}.png")
    readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
