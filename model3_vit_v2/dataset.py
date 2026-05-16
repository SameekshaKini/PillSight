"""
model3_vit/dataset.py
Heavy augmentation pipeline to fight overfitting on small dataset.
"""
import os, json
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import (
    FILTERED_CSV, CLASS_MAP_PATH, IMAGE_ROOT,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
    COLOR_JITTER, RANDOM_ERASING,
)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            # Multi-scale cropping — very effective for pills
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=COLOR_JITTER, contrast=COLOR_JITTER,
                saturation=COLOR_JITTER, hue=0.15
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=RANDOM_ERASING, scale=(0.02, 0.33)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


class PillDataset(Dataset):
    def __init__(self, df, class_to_idx, split="train"):
        self.df           = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform    = get_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(IMAGE_ROOT, row["image_path"])
        label = self.class_to_idx[str(row["label_code_id"])]
        img   = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_dataloaders(verbose=True):
    df = pd.read_csv(FILTERED_CSV)
    with open(CLASS_MAP_PATH) as f:
        class_to_idx = json.load(f)

    if verbose:
        print(f"[dataset] {len(df)} samples, {len(class_to_idx)} classes")
        print(f"[dataset] Label distribution:\n{df['label_code_id'].value_counts()}")

    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, stratify=df["label_code_id"], random_state=RANDOM_SEED
    )
    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio,
        stratify=train_val_df["label_code_id"], random_state=RANDOM_SEED
    )

    if verbose:
        print(f"[dataset] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_ds = PillDataset(train_df, class_to_idx, "train")
    val_ds   = PillDataset(val_df,   class_to_idx, "val")
    test_ds  = PillDataset(test_df,  class_to_idx, "test")

    # Weighted sampler for class imbalance
    labels       = [class_to_idx[str(r)] for r in train_df["label_code_id"]]
    class_counts = torch.bincount(torch.tensor(labels))
    weights      = 1.0 / class_counts.float()
    sample_wts   = weights[torch.tensor(labels)]
    sampler      = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_loader, val_loader, test_loader, class_to_idx
