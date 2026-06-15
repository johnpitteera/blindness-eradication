"""Fundus image preprocessing — pure cv2 / numpy, NO torch import.

This is deliberately decoupled from the deep-learning stack so it runs and tests
on any machine/Python version, independent of whether a torch wheel exists.

Pipeline (see `preprocess_image`):
    1. crop_retina      — remove the black border around the circular retina
    2. resize to square — fixed model input size
    3. circle_crop      — mask to a clean circle (kills corner artifacts)
    4. ben_graham       — subtract a local-average blur to amplify lesions and
                          normalize lighting across cameras

The disease signal in DR is small, low-contrast lesions (microaneurysms, exudates)
that camera/lighting variation easily washes out. Steps 1 and 4 are what make that
signal learnable; most silent preprocessing bugs live here, hence the tests.

References: Ben Graham's winning Kaggle DR 2015 preprocessing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "crop_retina",
    "circle_crop",
    "ben_graham",
    "preprocess_image",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

# ImageNet normalization stats — the pretrained backbone expects inputs scaled
# this way. Kept here (torch-free tuples) so both the dataset pipeline and the
# lightweight inference path can share them without importing the torch stack.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_retina(image: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop the black border surrounding the circular retina.

    Fundus photos sit on a black rectangle with the retina as a centered disc.
    We find the bounding box of pixels brighter than ``tol`` and crop to it.

    Args:
        image: HxWx3 uint8 BGR or RGB image (channel order preserved).
        tol:   intensity threshold; pixels with gray value > tol are "retina".

    Returns:
        Cropped image. If the image is essentially all black (degenerate), the
        original is returned unchanged rather than raising.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return image  # all-black / unreadable; let downstream quality check catch it

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image[y0:y1, x0:x1]
    # Guard against a 1-pixel-wide degenerate crop.
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        return image
    return cropped


def circle_crop(image: np.ndarray) -> np.ndarray:
    """Mask everything outside the largest inscribed circle to black.

    Applied after the image is square. Removes bright corner artifacts that the
    model could otherwise latch onto as a spurious shortcut.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(h, w) // 2
    cv2.circle(mask, center, radius, 255, thickness=-1)
    return cv2.bitwise_and(image, image, mask=mask)


def ben_graham(image: np.ndarray, sigma: float) -> np.ndarray:
    """Ben-Graham color normalization: amplify local contrast, flatten lighting.

    out = 4*image - 4*blur(image) + 128

    Subtracting a heavily-blurred copy removes the slowly-varying background
    illumination (which differs by camera), leaving high-frequency lesion detail
    centered around mid-gray.

    Args:
        image: HxWx3 uint8.
        sigma: Gaussian blur sigma in pixels. Larger sigma = gentler effect.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(image, 4, blurred, -4, 128)
    return out


def preprocess_image(
    image: np.ndarray,
    image_size: int = 512,
    ben_graham_enabled: bool = True,
    ben_graham_sigma_scale: float = 10.0,
    crop_tol: int = 7,
) -> np.ndarray:
    """Full preprocessing pipeline for one fundus image.

    Args:
        image:                 HxWx3 uint8 RGB image (as read by PIL/np, not BGR).
        image_size:            output square side length.
        ben_graham_enabled:    apply Ben-Graham normalization.
        ben_graham_sigma_scale: sigma = image_size / this.
        crop_tol:              threshold for `crop_retina`.

    Returns:
        image_size x image_size x 3 uint8 RGB image, ready for tensor conversion.
    """
    img = crop_retina(image, tol=crop_tol)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = circle_crop(img)
    if ben_graham_enabled:
        sigma = image_size / float(ben_graham_sigma_scale)
        img = ben_graham(img, sigma=sigma)
        # Re-mask: addWeighted's +128 bias lifts the black corners off zero.
        img = circle_crop(img)
    return img


# --------------------------------------------------------------------------- #
# CLI: visualize the pipeline on a real image so you can eyeball it locally.
# --------------------------------------------------------------------------- #
def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _main() -> None:
    p = argparse.ArgumentParser(description="Visualize DR fundus preprocessing.")
    p.add_argument("--input", required=True, type=Path, help="Path to a fundus image.")
    p.add_argument("--output", default=Path("preprocessed.png"), type=Path)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--no-ben-graham", action="store_true")
    args = p.parse_args()

    rgb = _read_rgb(args.input)
    out = preprocess_image(
        rgb,
        image_size=args.image_size,
        ben_graham_enabled=not args.no_ben_graham,
    )
    # Save side-by-side original (resized) vs processed for comparison.
    orig_resized = cv2.resize(rgb, (args.image_size, args.image_size))
    combo = np.hstack([orig_resized, out])
    cv2.imwrite(str(args.output), cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
    print(f"Wrote comparison (original | processed) to {args.output}")


if __name__ == "__main__":
    _main()
