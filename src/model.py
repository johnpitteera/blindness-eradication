"""Model definition — timm EfficientNet backbone + DR head.

Two head formulations:
  * 'classification' (v1 default): a single Linear -> 5 logits, trained with
    (weighted) cross-entropy. Predict via argmax.
  * 'ordinal' (scaffolded): K-1 = 4 sigmoid outputs encoding P(grade > k).
    Respects the natural order of DR grades; often improves QWK. Trained with
    BCE against a cumulative target. Predict by counting outputs > 0.5.

v1 should be validated end-to-end on the classification head before relying on
the ordinal path.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class DRModel(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        head: str = "classification",
    ):
        super().__init__()
        self.head_type = head
        self.num_classes = num_classes

        # num_classes=0 -> timm returns pooled features, we attach our own head.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features

        out_dim = num_classes if head == "classification" else num_classes - 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    @torch.no_grad()
    def predict_grades(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw outputs to integer grades 0..num_classes-1."""
        out = self.forward(x)
        if self.head_type == "classification":
            return out.argmax(dim=1)
        # ordinal: count how many cumulative thresholds are exceeded.
        probs = torch.sigmoid(out)
        return (probs > 0.5).sum(dim=1)


def ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer labels to cumulative binary targets for ordinal head.

    grade 2, num_classes 5 -> [1, 1, 0, 0]  (i.e. grade>0, grade>1, grade>2, grade>3)
    """
    k = num_classes - 1
    rng = torch.arange(k, device=labels.device).unsqueeze(0)  # (1, k)
    return (labels.unsqueeze(1) > rng).float()  # (B, k)


def build_loss(cfg, class_weights: torch.Tensor | None, device):
    """Pick the loss matching config.

    weighted_ce / ce -> CrossEntropyLoss (classification head)
    mse              -> MSELoss on the single-logit regression variant (advanced)
    For an ordinal head, BCEWithLogitsLoss is used regardless of `loss` name.
    """
    head = cfg.model.head
    if head == "ordinal":
        return nn.BCEWithLogitsLoss()

    loss_name = cfg.training.loss
    if loss_name == "weighted_ce":
        w = class_weights.to(device) if class_weights is not None else None
        return nn.CrossEntropyLoss(weight=w)
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss '{loss_name}' for head '{head}'.")


def build_model(cfg, device) -> DRModel:
    model = DRModel(
        backbone=cfg.model.backbone,
        num_classes=cfg.project.num_classes,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
        head=cfg.model.head,
    )
    return model.to(device)
