"""
model3_vit/model.py
ViT-B/16 fine-tuned for pill classification.

Strategy:
  Phase 1 (epochs 1..UNFREEZE_EPOCH):  Only the new head trains. LR = HEAD_LR.
  Phase 2 (epochs UNFREEZE_EPOCH+1..): Full model trains. LR = FINETUNE_LR.
"""
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from config import NUM_CLASSES, PRETRAINED


class ViTPillClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if PRETRAINED else None
        self.vit = vit_b_16(weights=weights)

        # Replace the head with a simple linear layer — simpler is better
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, NUM_CLASSES),
        )

    def forward(self, x):
        return self.vit(x)

    def freeze_backbone(self):
        """Freeze everything except heads."""
        for name, param in self.vit.named_parameters():
            param.requires_grad = "heads" in name

    def unfreeze_all(self):
        """Unfreeze everything."""
        for param in self.parameters():
            param.requires_grad = True

    def param_groups(self, head_lr, backbone_lr):
        """Return parameter groups with different LRs."""
        head_params     = [p for n, p in self.named_parameters() if "heads" in n]
        backbone_params = [p for n, p in self.named_parameters() if "heads" not in n]
        return [
            {"params": head_params,     "lr": head_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]

    def stats(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(device):
    model = ViTPillClassifier().to(device)
    model.freeze_backbone()
    total, trainable = model.stats()
    print(f"[model] ViT-B/16  total={total:,}  trainable={trainable:,}")
    return model
