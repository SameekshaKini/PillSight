"""
model3_vit/model.py
Vision Transformer (ViT-B/16) adapted for pill classification.

Strategy
--------
1. Load ViT-B/16 pretrained on ImageNet-21k (via torchvision).
2. Replace the classification head with a two-layer MLP for NUM_CLASSES.
3. Initially freeze the backbone; unfreeze at UNFREEZE_EPOCH for fine-tuning.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from config import NUM_CLASSES, PRETRAINED


class ViTPillClassifier(nn.Module):
    """ViT-B/16 with a custom two-layer MLP head."""

    def __init__(self):
        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1 if PRETRAINED else None
        self.vit = vit_b_16(weights=weights)

        # ViT-B/16 head is vit.heads.head — replace it
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.vit(x)

    # ── Freeze / unfreeze helpers ──────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze everything except the classification head."""
        for name, param in self.vit.named_parameters():
            if "heads" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = True

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(device):
    model = ViTPillClassifier()
    model.freeze_backbone()          # start frozen; unfrozen at UNFREEZE_EPOCH
    model = model.to(device)

    total, trainable = model.count_params()
    print(f"[model] ViT-B/16  total={total:,}  trainable={trainable:,}")
    return model
