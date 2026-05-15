"""
model3_vit/dataset.py
PyTorch Dataset for the ePillID segmented reference images.
Mirrors the interface used by model1_cnn/dataset.py so scripts stay consistent.
"""

import os
import json
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import (
    FILTERED_CSV, CLASS_MAP_PATH, IMAGE_ROOT,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
    COLOR_JITTER, RAND_ERASING,
)


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_transforms(split: str):
    """
    Returns the appropriate torchvision transform pipeline.
    ViT uses the same ImageNet mean/std as the pretrained weights.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=COLOR_JITTER,
                contrast=COLOR_JITTER,
                saturation=COLOR_JITTER,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=RAND_ERASING),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Dataset class ──────────────────────────────────────────────────────────────

class PillDataset(Dataset):
    """
    Loads pill images from filtered_metadata.csv.
    Expects columns: image_path (relative to IMAGE_ROOT), label_code_id.
    """

    def __init__(self, df: pd.DataFrame, class_to_idx: dict, split: str = "train"):
        self.df           = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform    = get_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(IMAGE_ROOT, row["image_path"])
        label = self.class_to_idx[str(row["label_code_id"])]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders():
    """
    Reads filtered_metadata.csv, splits train/val/test, returns three DataLoaders
    plus the class_to_idx mapping.
    """
    df = pd.read_csv(FILTERED_CSV)

    with open(CLASS_MAP_PATH) as f:
        class_to_idx = json.load(f)   # {"label_code_id_str": int_index, ...}

    # Reproducible stratified split
    from sklearn.model_selection import train_test_split

    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, stratify=df["label_code_id"], random_state=RANDOM_SEED
    )
    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, stratify=train_val_df["label_code_id"],
        random_state=RANDOM_SEED
    )

    train_ds = PillDataset(train_df, class_to_idx, split="train")
    val_ds   = PillDataset(val_df,   class_to_idx, split="val")
    test_ds  = PillDataset(test_df,  class_to_idx, split="test")

    # Weighted sampler to handle class imbalance during training
    labels = [class_to_idx[str(r)] for r in train_df["label_code_id"]]
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts.float()
    sample_weights = weights[torch.tensor(labels)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    return train_loader, val_loader, test_loader, class_to_idx
