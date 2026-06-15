"""Load and validate config.yaml into a typed, attribute-access object.

Kept dependency-light (only pyyaml) so it can be imported anywhere, including
by scripts that never touch torch.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(obj: Any) -> Any:
    """Recursively turn nested dicts into attribute-accessible namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path = "config.yaml") -> SimpleNamespace:
    """Read config.yaml and return it as nested namespaces.

    Access like: ``cfg.training.epochs``, ``cfg.paths.images_dir``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at '{path}'. Run from the project root or pass --config."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _validate(raw)
    cfg = _to_namespace(raw)
    # Keep the raw dict around for serialization / logging.
    cfg._raw = raw  # type: ignore[attr-defined]
    return cfg


def _validate(raw: dict) -> None:
    """Fail fast on the few invariants the rest of the code relies on."""
    required_top = ["project", "paths", "preprocessing", "model", "training"]
    missing = [k for k in required_top if k not in raw]
    if missing:
        raise ValueError(f"config.yaml missing required sections: {missing}")

    if raw["project"]["num_classes"] != 5:
        # The clinical ICDR scale is 5-class; QWK weighting assumes this.
        raise ValueError("project.num_classes must be 5 for the ICDR DR scale.")

    head = raw["model"].get("head", "classification")
    if head not in ("classification", "ordinal"):
        raise ValueError(f"model.head must be 'classification' or 'ordinal', got '{head}'.")

    loss = raw["training"].get("loss", "weighted_ce")
    if loss not in ("weighted_ce", "ce", "mse"):
        raise ValueError(f"training.loss must be one of weighted_ce|ce|mse, got '{loss}'.")


if __name__ == "__main__":
    import json

    cfg = load_config()
    print(json.dumps(cfg._raw, indent=2))
