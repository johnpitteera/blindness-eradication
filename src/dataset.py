"""PyTorch Dataset + dataloaders for APTOS-style DR data.

Imports torch — runs on Colab/GPU or locally (CPU torch wheel). The heavy image
work is delegated to `preprocessing.preprocess_image` (torch-free), so the
preprocessing logic stays testable without this module.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess_image


def build_transforms(cfg, train: bool) -> A.Compose:
    """Augmentation + normalization. Augment only on train.

    Geometric augments are safe and effective here: the retina has no canonical
    orientation, so full rotation and flips create valid new views.
    """
    norm = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    if not train:
        return A.Compose(norm)

    aug = cfg.augmentation
    return A.Compose(
        [
            A.HorizontalFlip(p=aug.horizontal_flip),
            A.VerticalFlip(p=aug.vertical_flip),
            A.Rotate(limit=aug.rotate_limit, border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=aug.brightness_contrast,
                contrast_limit=aug.brightness_contrast,
                p=0.5,
            ),
            *norm,
        ]
    )


class FundusDataset(Dataset):
    """Maps (id_code, diagnosis) rows to (preprocessed_tensor, label).

    Args:
        df:         DataFrame with columns id_code, diagnosis.
        images_dir: folder containing <id_code><image_ext> files.
        cfg:        loaded config namespace.
        train:      whether to apply training augmentation.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, cfg, train: bool):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.cfg = cfg
        self.image_ext = cfg.data.image_ext
        self.transforms = build_transforms(cfg, train=train)
        self.pre = cfg.preprocessing

    def __len__(self) -> int:
        return len(self.df)

    def _load_rgb(self, id_code: str) -> np.ndarray:
        path = self.images_dir / f"{id_code}{self.image_ext}"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rgb = self._load_rgb(str(row["id_code"]))
        img = preprocess_image(
            rgb,
            image_size=self.pre.image_size,
            ben_graham_enabled=self.pre.ben_graham,
            ben_graham_sigma_scale=self.pre.ben_graham_sigma_scale,
            crop_tol=self.pre.crop_tol,
        )
        tensor = self.transforms(image=img)["image"]
        label = int(row["diagnosis"])
        return tensor, label


def make_splits(cfg, seed: int):
    """Stratified train/val split of the labels CSV.

    Stratifying by grade keeps the (imbalanced) class ratios consistent across
    splits, so validation QWK reflects real performance.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(cfg.paths.labels_csv)
    if not {"id_code", "diagnosis"}.issubset(df.columns):
        raise ValueError(
            f"labels_csv must have columns id_code,diagnosis; got {list(df.columns)}"
        )
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.data.val_split,
        random_state=seed,
        stratify=df["diagnosis"],
    )
    return train_df, val_df


def class_weights(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for weighted cross-entropy.

    Most fundus images are grade 0; without this the model collapses to always
    predicting "no DR". Weights are normalized to mean 1.0.
    """
    counts = np.bincount(df["diagnosis"].to_numpy(), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)  # avoid div-by-zero on absent classes
    inv = counts.sum() / counts
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def build_dataloaders(cfg, seed: int):
    """Returns (train_loader, val_loader, class_weights_tensor)."""
    train_df, val_df = make_splits(cfg, seed)
    train_ds = FundusDataset(train_df, cfg.paths.images_dir, cfg, train=True)
    val_ds = FundusDataset(val_df, cfg.paths.images_dir, cfg, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    weights = class_weights(train_df, cfg.project.num_classes)
    return train_loader, val_loader, weights
