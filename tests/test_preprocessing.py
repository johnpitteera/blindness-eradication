"""Tests for src/preprocessing.py — runs without torch.

These guard the bugs that silently ruin a DR model: off-center / wrong-size crops,
channel-order mistakes, and the circle mask not actually zeroing the corners.
"""
import numpy as np
import pytest

from src.preprocessing import (
    ben_graham,
    circle_crop,
    crop_retina,
    preprocess_image,
)


def _fake_fundus(size: int = 600, border: int = 80) -> np.ndarray:
    """A synthetic fundus: bright disc centered on a black rectangle."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    radius = (size // 2) - border
    yy, xx = np.ogrid[:size, :size]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    img[disc] = (180, 90, 60)  # RGB-ish retina tone
    return img


def test_crop_retina_removes_black_border():
    img = _fake_fundus(size=600, border=80)
    cropped = crop_retina(img, tol=7)
    # The disc has diameter 2*(300-80)=440; crop should be ~that, not the full 600.
    assert cropped.shape[0] < img.shape[0]
    assert cropped.shape[1] < img.shape[1]
    assert 400 <= cropped.shape[0] <= 460
    # Cropped region should be mostly non-black now.
    assert (cropped.sum(axis=2) > 0).mean() > 0.6


def test_crop_retina_handles_all_black():
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    out = crop_retina(black)
    # Degenerate input returns unchanged rather than raising.
    assert out.shape == black.shape


def test_crop_retina_rejects_bad_shape():
    with pytest.raises(ValueError):
        crop_retina(np.zeros((100, 100), dtype=np.uint8))  # missing channel dim


def test_circle_crop_zeros_corners():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    out = circle_crop(img)
    # Corners must be black; center must be preserved.
    assert out[0, 0].sum() == 0
    assert out[-1, -1].sum() == 0
    assert out[50, 50].sum() > 0


def test_ben_graham_centers_around_mid_gray():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    out = ben_graham(img, sigma=6.0)
    # Flat input -> 4*x - 4*blur(x) + 128 ≈ 128 everywhere.
    assert abs(int(out.mean()) - 128) <= 2
    assert out.dtype == np.uint8


def test_preprocess_image_output_contract():
    img = _fake_fundus()
    out = preprocess_image(img, image_size=256, ben_graham_enabled=True)
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8
    # Circle mask is applied last, so corners must be black.
    assert out[0, 0].sum() == 0


def test_preprocess_image_without_ben_graham():
    img = _fake_fundus()
    out = preprocess_image(img, image_size=128, ben_graham_enabled=False)
    assert out.shape == (128, 128, 3)
    assert out[0, 0].sum() == 0  # corners still masked
