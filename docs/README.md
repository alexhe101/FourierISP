# Project website

The website is a standalone static research page (no build dependencies or external scripts).

Preview from the repository root:

```sh
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. To publish on GitHub Pages, open [repository Pages settings](https://github.com/alexhe101/FourierISP/settings/pages) and select **Deploy from a branch → main → /docs → Save**. The expected public URL is https://alexhe101.github.io/FourierISP/.

The schematic in assets/architecture.svg is an explanatory redrawing, not a model output. All quantitative results are from the original AAAI 2024 paper.

## Original paper result panels

The page references Figures 1, 5–8 of arXiv:2401.02161v1 and includes the Table 1 comparisons and Table 2 ablation. To archive the original images locally:

```sh
python -m pip install requests Pillow PyMuPDF
python scripts/fetch_paper_figures.py
```

The dedicated GitHub Actions workflow also retrieves and archives them in `docs/assets/paper/`. It uses only the public paper and does not run inference. If a remote image cannot load, the page provides a PDF link. A bot commit may not trigger branch-based Pages deployment; after extraction, redeploy Pages if necessary.
