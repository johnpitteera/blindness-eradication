"""Single-image inference: fundus image -> DR grade + class probabilities.

    python -m src.inference --checkpoint outputs/best_model.pth --image fundus.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import load_config
from .model import build_model
from .preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocess_image

GRADE_NAMES = {
    0: "No DR",
    1: "Mild non-proliferative",
    2: "Moderate non-proliferative",
    3: "Severe non-proliferative",
    4: "Proliferative DR",
}


def _load_checkpoint(checkpoint_path: str, config_path: str, device):
    """Rebuild the model from the config embedded in the checkpoint when present."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        # Write the embedded config to load via the normal path.
        from .config import _to_namespace

        cfg = _to_namespace(ckpt["config"])
        cfg._raw = ckpt["config"]
        state = ckpt["model_state"]
    else:
        cfg = load_config(config_path)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

    model = build_model(cfg, device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def preprocess_to_tensor(image_path: str, cfg) -> torch.Tensor:
    """Image path -> normalized (1,3,H,W) tensor matching training."""
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = preprocess_image(
        rgb,
        image_size=cfg.preprocessing.image_size,
        ben_graham_enabled=cfg.preprocessing.ben_graham,
        ben_graham_sigma_scale=cfg.preprocessing.ben_graham_sigma_scale,
        crop_tol=cfg.preprocessing.crop_tol,
    ).astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def _tta_views(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Original + horizontal/vertical/180 flips.

    The retina has no canonical orientation, so flips are label-preserving.
    Averaging predictions over them reduces variance and typically nudges QWK up.
    """
    return [
        tensor,
        torch.flip(tensor, dims=[3]),   # horizontal
        torch.flip(tensor, dims=[2]),   # vertical
        torch.flip(tensor, dims=[2, 3]),  # 180 degrees
    ]


@torch.no_grad()
def predict(model, tensor: torch.Tensor, device, head: str, tta: bool = False) -> dict:
    tensor = tensor.to(device)
    views = _tta_views(tensor) if tta else [tensor]

    if head == "classification":
        probs = torch.stack(
            [torch.softmax(model(v), dim=1)[0] for v in views]
        ).mean(dim=0).cpu().numpy()
        grade = int(probs.argmax())
        confidence = float(probs[grade])
        prob_list = probs.tolist()
    else:  # ordinal
        sig = torch.stack(
            [torch.sigmoid(model(v))[0] for v in views]
        ).mean(dim=0).cpu().numpy()
        grade = int((sig > 0.5).sum())
        confidence = float(np.mean(np.abs(sig - 0.5) * 2))  # rough certainty proxy
        prob_list = sig.tolist()
    return {
        "grade": grade,
        "label": GRADE_NAMES.get(grade, str(grade)),
        "confidence": confidence,
        "referable": grade >= 2,
        "raw": prob_list,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Predict DR grade for one fundus image.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--tta", action="store_true",
                   help="Test-time augmentation: average over flips for a steadier prediction.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_checkpoint(args.checkpoint, args.config, device)
    tensor = preprocess_to_tensor(args.image, cfg)
    result = predict(model, tensor, device, cfg.model.head, tta=args.tta)

    print(f"Image:      {Path(args.image).name}")
    print(f"Grade:      {result['grade']}  ({result['label']})")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Referable:  {'YES — refer to specialist' if result['referable'] else 'no'}")
    if cfg.model.head == "classification":
        print("Class probabilities:")
        for i, prob in enumerate(result["raw"]):
            print(f"  {i} {GRADE_NAMES[i]:<28} {prob:.3f}")


if __name__ == "__main__":
    main()
