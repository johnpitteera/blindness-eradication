# Data — APTOS 2019 Blindness Detection

The default dataset is the **APTOS 2019** competition data: ~3,662 training fundus
images, each graded 0–4. It's the right size for free Colab GPU training.

## Expected layout

After download, this folder should look like:

```
data/
└── aptos2019/
    ├── train.csv            # columns: id_code,diagnosis
    └── train_images/        # <id_code>.png  files
```

These paths match `config.yaml`. If you put data elsewhere, edit `paths:` there.

## Option A — Kaggle CLI (recommended, works on Colab)

APTOS is **Kaggle-gated**: you need a Kaggle account, an API token, and you must
accept the competition rules once on the website.

1. Create an account at https://www.kaggle.com
2. Open the competition page and click **"Join / Accept rules"**:
   https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules
   (Without this, downloads 403 even with a valid token.)
3. Get an API token: Kaggle → your profile → **Settings → API → Create New Token**.
   This downloads `kaggle.json` (contains your username + key).
4. Install and authenticate the CLI:

   ```bash
   pip install kaggle
   mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/    # Linux/Colab
   chmod 600 ~/.kaggle/kaggle.json
   ```

   On Windows (local), put `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`.

5. Download and unzip into place:

   ```bash
   kaggle competitions download -c aptos2019-blindness-detection -p data/aptos2019
   cd data/aptos2019 && unzip -q aptos2019-blindness-detection.zip
   ```

   APTOS images are `.png`. If you use a dataset with `.jpg`, set
   `data.image_ext: .jpg` in `config.yaml`.

## Option B — Colab secret (cleanest for the notebook)

In Colab, store your Kaggle key via the **🔑 Secrets** panel as `KAGGLE_USERNAME`
and `KAGGLE_KEY`. The notebook reads them and runs the download for you — no file
juggling. See `notebooks/colab_train.ipynb`.

## Other datasets you can plug in

The pipeline only needs `(image_path, integer_grade_0_to_4)` pairs, so any of
these work by adjusting `paths:` and the CSV columns in `src/dataset.py`:

- **EyePACS / Kaggle DR 2015** — ~88k images, larger & noisier (needs more compute).
- **Messidor-2** — clean, clinically graded; great as an *external* validation set.
- **IDRiD** — adds pixel-level lesion masks (for future segmentation work).

> Do **not** commit the image data or `kaggle.json` to git. Add `data/aptos2019/`
> and `*.json` to `.gitignore`.
