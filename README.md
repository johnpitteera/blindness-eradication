# Blindness Eradication — Diabetic Retinopathy Detection

[![CI](https://github.com/johnpitteera/blindness-eradication/actions/workflows/ci.yml/badge.svg)](https://github.com/johnpitteera/blindness-eradication/actions/workflows/ci.yml)

Deep-learning pipeline that grades **diabetic retinopathy (DR)** from fundus
(retina) photographs on the 5-class international clinical scale:

| Grade | Meaning                     |
|-------|-----------------------------|
| 0     | No DR                       |
| 1     | Mild non-proliferative      |
| 2     | Moderate non-proliferative  |
| 3     | Severe non-proliferative    |
| 4     | Proliferative DR            |

Built with **PyTorch** + **timm** (EfficientNet backbone). Developed locally,
trained on a GPU (Google Colab recommended — see `notebooks/colab_train.ipynb`).

---

## Why these choices

- **Metric: Quadratic Weighted Kappa (QWK)**, not accuracy. The classes are
  imbalanced (most images are grade 0) and *ordered* (predicting 0 for a grade-4
  patient is far worse than predicting 3). QWK rewards being close. We also track
  **sensitivity** because missing a sick patient is the costly error in screening.
- **Backbone: EfficientNet-B3** via `timm`, pretrained on ImageNet (transfer
  learning). There isn't enough retinal data to train from scratch.
- **Dataset: APTOS 2019** (~3,600 graded images) — small enough for free Colab,
  clinically graded. See [data/README.md](data/README.md).

## Project layout

```
.
├── config.yaml              # all paths + hyperparameters in one place
├── requirements.txt         # local (CPU) deps; Colab keeps its own torch
├── data/README.md           # how to download APTOS via Kaggle
├── src/
│   ├── config.py            # load + validate config.yaml
│   ├── preprocessing.py     # retina crop, resize, Ben-Graham color norm (pure cv2/numpy)
│   ├── dataset.py           # PyTorch Dataset + albumentations augmentation
│   ├── model.py             # EfficientNet + 5-class head (ordinal option scaffolded)
│   ├── train.py             # training loop, weighted loss, QWK, checkpointing
│   ├── evaluate.py          # QWK, sensitivity/specificity, confusion matrix
│   ├── inference.py         # predict a single image
│   └── gradcam.py           # Grad-CAM explainability overlays
├── notebooks/colab_train.ipynb
└── tests/test_preprocessing.py
```

## Quick start (local, CPU — for development & smoke tests)

```powershell
python -m pip install -r requirements.txt

# 1. Sanity-check preprocessing on one image (no torch needed):
python -m pytest tests/ -v

# 2. Visualize preprocessing on a sample fundus image:
python -m src.preprocessing --input path\to\fundus.jpg --output out.png

# 3. Smoke-test the whole training pipeline on tiny synthetic data:
python -m src.train --config config.yaml --smoke-test
```

## Model variants & options

- **Ordinal head** — respects the natural order of DR grades (often better QWK).
  Set `model.head: ordinal` in `config.yaml`, or override per-run:
  `python -m src.train --head ordinal`
- **LR warmup** — linear warmup before cosine decay, stabilizes the pretrained
  backbone. Tune `training.warmup_epochs` (0 disables).
- **Test-time augmentation** — average predictions over flips for a steadier
  result: `python -m src.inference --checkpoint ... --image ... --tta`

## Real training (Colab GPU)

Open `notebooks/colab_train.ipynb` in Google Colab, set the runtime to **GPU**,
follow the cells. Colab already ships a CUDA build of PyTorch — the notebook
installs only the *extra* packages so it doesn't clobber the GPU torch.

## Important caveats

This is a **screening / decision-support** tool, **not** a diagnostic device. It
must not be used for real clinical decisions without regulatory clearance
(FDA/CE), prospective validation, and clinician oversight. Models trained on one
population/camera often degrade on another — always validate externally.
