"""Grad-CAM explainability: overlay a heatmap showing which retinal regions
drove the model's grade.

Two purposes:
  1. Clinical trust — a clinician can see the model attended to actual lesions.
  2. Debugging — if the heatmap lights up the image border or a camera artifact
     instead of the retina, the model learned a shortcut and you can't trust it.

    python -m src.gradcam --checkpoint outputs/best_model.pth --image fundus.jpg \
        --output cam.png

Uses the `grad-cam` package if installed; otherwise falls back to a small
built-in Grad-CAM so the module still works with only torch.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from .inference import _load_checkpoint, preprocess_to_tensor


def _find_target_layer(model) -> torch.nn.Module:
    """Last conv layer of the timm backbone — the standard Grad-CAM target."""
    last_conv = None
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last_conv


class _SimpleGradCAM:
    """Minimal Grad-CAM (fallback when the grad-cam package isn't present)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, output):
        self.activations = output.detach()

    def _bwd(self, _m, _gi, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)[0, 0].cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def generate_cam(model, tensor: torch.Tensor, device) -> np.ndarray:
    """Return a normalized [0,1] HxW class-activation map for the top class."""
    model.eval()
    tensor = tensor.to(device).requires_grad_(True)
    target_layer = _find_target_layer(model)
    cam = _SimpleGradCAM(model, target_layer)(tensor)
    return cam


def overlay_cam(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend the heatmap over the original image (both resized to match)."""
    h, w = image_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Grad-CAM overlay for a fundus prediction.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="gradcam.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_checkpoint(args.checkpoint, args.config, device)
    tensor = preprocess_to_tensor(args.image, cfg)
    cam = generate_cam(model, tensor, device)

    original = cv2.imread(args.image, cv2.IMREAD_COLOR)
    original = cv2.resize(original, (cfg.preprocessing.image_size,) * 2)
    overlay = overlay_cam(original, cam)
    cv2.imwrite(args.output, overlay)
    print(f"Wrote Grad-CAM overlay to {args.output}")


if __name__ == "__main__":
    main()
