"""Evaluation metrics for DR grading.

Headline metric is Quadratic Weighted Kappa (QWK) — the clinical standard for
ordered grading. We also report per-class sensitivity/specificity and the
"referable DR" sensitivity (grade >= 2), because in screening a false negative
(missing a sick patient) is the costly error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
)

# Grades >= 2 (moderate+) are "referable" — should be seen by a specialist.
REFERABLE_THRESHOLD = 2


@dataclass
class EvalReport:
    qwk: float
    accuracy: float
    referable_sensitivity: float
    referable_specificity: float
    per_class_sensitivity: list[float]
    confusion: list[list[int]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"QWK (quadratic weighted kappa): {self.qwk:.4f}",
            f"Accuracy:                       {self.accuracy:.4f}",
            f"Referable DR sensitivity (>=2): {self.referable_sensitivity:.4f}",
            f"Referable DR specificity:       {self.referable_specificity:.4f}",
            "Per-class sensitivity (recall): "
            + ", ".join(f"{i}:{s:.3f}" for i, s in enumerate(self.per_class_sensitivity)),
        ]
        return "\n".join(lines)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5
) -> EvalReport:
    """Compute the full report from integer grade arrays."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(num_classes))

    qwk = float(
        cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)
    )
    accuracy = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class sensitivity = TP / (actual positives for that class).
    per_class_sens = []
    for c in labels:
        actual_c = cm[c, :].sum()
        per_class_sens.append(float(cm[c, c] / actual_c) if actual_c > 0 else 0.0)

    # Binarize at the referable threshold for screening-relevant sens/spec.
    true_ref = y_true >= REFERABLE_THRESHOLD
    pred_ref = y_pred >= REFERABLE_THRESHOLD
    tp = int((true_ref & pred_ref).sum())
    fn = int((true_ref & ~pred_ref).sum())
    tn = int((~true_ref & ~pred_ref).sum())
    fp = int((~true_ref & pred_ref).sum())
    ref_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ref_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return EvalReport(
        qwk=qwk,
        accuracy=accuracy,
        referable_sensitivity=ref_sens,
        referable_specificity=ref_spec,
        per_class_sensitivity=per_class_sens,
        confusion=cm.tolist(),
    )


def save_confusion_plot(cm: list[list[int]], out_path: str, num_classes: int = 5) -> None:
    """Save a labeled confusion-matrix heatmap (best-effort; skips if no mpl)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
