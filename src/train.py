"""Training entry point.

    python -m src.train --config config.yaml                # real training
    python -m src.train --config config.yaml --smoke-test   # tiny synthetic run

The smoke test fabricates random tensors so the full loop (forward, loss,
backward, validation, QWK, checkpointing) can be exercised with no dataset and
on CPU — that's how we prove the plumbing before spending GPU time on APTOS.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import load_config
from .evaluate import compute_metrics, save_confusion_plot
from .model import build_loss, build_model, ordinal_targets


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scheduler(optimizer, cfg):
    """Linear LR warmup for `warmup_epochs`, then cosine decay over the rest.

    Warmup avoids destabilizing the pretrained backbone with large early steps;
    cosine decay anneals smoothly toward the end. Falls back to plain cosine when
    warmup_epochs is 0 or there aren't enough epochs to warm up.
    """
    total = max(1, cfg.training.epochs)
    warmup = int(getattr(cfg.training, "warmup_epochs", 0) or 0)
    warmup = max(0, min(warmup, total - 1))  # leave at least 1 epoch for decay

    cosine_epochs = max(1, total - warmup)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    if warmup == 0:
        return cosine

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup]
    )


def _synthetic_loaders(cfg, device):
    """Random-tensor loaders mirroring the real data contract, for --smoke-test."""
    st = cfg.smoke_test
    n, size, nc = st.num_samples, st.image_size, cfg.project.num_classes
    g = torch.Generator().manual_seed(cfg.project.seed)
    x = torch.randn(n, 3, size, size, generator=g)
    y = torch.randint(0, nc, (n,), generator=g)
    ds = TensorDataset(x, y)
    n_val = max(st.batch_size, n // 5)
    train_ds = torch.utils.data.Subset(ds, range(n - n_val))
    val_ds = torch.utils.data.Subset(ds, range(n - n_val, n))
    train_loader = DataLoader(train_ds, batch_size=st.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=st.batch_size)
    weights = torch.ones(nc)
    return train_loader, val_loader, weights


def _apply_smoke_overrides(cfg) -> None:
    """Shrink the model/run so the synthetic loop finishes in seconds on CPU.

    Mutates both the namespace AND cfg._raw so a smoke checkpoint records the
    model it actually contains (resnet18), keeping inference/gradcam loadable.
    """
    st = cfg.smoke_test
    cfg.training.epochs = st.epochs
    cfg.training.batch_size = st.batch_size
    cfg.training.mixed_precision = False
    cfg.model.backbone = "resnet18"   # tiny, fast to instantiate
    cfg.model.pretrained = False      # no network download during a smoke test

    cfg._raw["training"].update(
        {"epochs": st.epochs, "batch_size": st.batch_size, "mixed_precision": False}
    )
    cfg._raw["model"].update({"backbone": "resnet18", "pretrained": False})


def _compute_loss(criterion, outputs, labels, cfg):
    if cfg.model.head == "ordinal":
        targets = ordinal_targets(labels, cfg.project.num_classes)
        return criterion(outputs, targets)
    if cfg.training.loss == "mse":  # regression variant expects float targets
        return criterion(outputs.squeeze(-1), labels.float())
    return criterion(outputs, labels)


@torch.no_grad()
def validate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        grades = model.predict_grades(x)
        preds.append(grades.cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def train(cfg, smoke: bool = False) -> dict:
    set_seed(cfg.project.seed)
    device = get_device()
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        _apply_smoke_overrides(cfg)
        train_loader, val_loader, weights = _synthetic_loaders(cfg, device)
        print(f"[smoke-test] device={device}  backbone={cfg.model.backbone}")
    else:
        from .dataset import build_dataloaders  # imports cv2/albumentations lazily

        train_loader, val_loader, weights = build_dataloaders(cfg, cfg.project.seed)
        print(f"device={device}  backbone={cfg.model.backbone}  "
              f"train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    model = build_model(cfg, device)
    criterion = build_loss(cfg, weights, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = bool(cfg.training.mixed_precision) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_qwk = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(x)
                loss = _compute_loss(criterion, outputs, y, cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        train_loss = running / len(train_loader.dataset)
        y_true, y_pred = validate(model, val_loader, device)
        report = compute_metrics(y_true, y_pred, cfg.project.num_classes)
        dt = time.time() - t0
        print(
            f"epoch {epoch:>2}/{cfg.training.epochs}  loss={train_loss:.4f}  "
            f"val_QWK={report.qwk:.4f}  ref_sens={report.referable_sensitivity:.3f}  "
            f"({dt:.1f}s)"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "qwk": report.qwk})

        if report.qwk > best_qwk:
            best_qwk = report.qwk
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(
                {"model_state": best_state, "config": cfg._raw, "qwk": best_qwk},
                out_dir / "best_model.pth",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no QWK gain for "
                      f"{epochs_no_improve} epochs).")
                break

    # Final report from the best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_pred = validate(model, val_loader, device)
    final = compute_metrics(y_true, y_pred, cfg.project.num_classes)
    print("\n=== best model ===")
    print(final.summary())

    metrics = {
        "best_qwk": best_qwk,
        "final": {
            "qwk": final.qwk,
            "accuracy": final.accuracy,
            "referable_sensitivity": final.referable_sensitivity,
            "referable_specificity": final.referable_specificity,
            "per_class_sensitivity": final.per_class_sensitivity,
            "confusion": final.confusion,
        },
        "history": history,
        "smoke_test": smoke,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_plot(final.confusion, str(out_dir / "confusion.png"),
                        cfg.project.num_classes)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train the DR grading model.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny synthetic run to validate the pipeline on CPU.")
    p.add_argument("--head", choices=["classification", "ordinal"], default=None,
                   help="Override model.head from the config.")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.head is not None:
        cfg.model.head = args.head
        cfg._raw["model"]["head"] = args.head
    train(cfg, smoke=args.smoke_test)


if __name__ == "__main__":
    main()
