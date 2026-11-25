# Project website

The website is a standalone static research page (no build dependencies or external scripts).

Preview from the repository root:

```sh
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. To publish on GitHub Pages, open [repository Pages settings](https://github.com/alexhe101/FourierISP/settings/pages) and select **Deploy from a branch → main → /docs → Save**. The expected public URL is https://alexhe101.github.io/FourierISP/.

The schematic in assets/architecture.svg is an explanatory redrawing, not a model output. All quantitative results are from the original AAAI 2024 paper.
