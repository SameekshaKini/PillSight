import torch.nn as nn
import torchvision.models as models

def build_vit(num_classes: int, freeze_backbone: bool = True):
    """
    Load pretrained ViT-B/16 and replace the classification head.
    freeze_backbone=True  → only head trains (phase 1)
    freeze_backbone=False → full model trains (phase 2)
    """
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    # Replace the head
    in_features = vit.heads.head.in_features
    vit.heads.head = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in vit.named_parameters():
            if "heads" not in name:
                param.requires_grad = False

    return vit

def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True