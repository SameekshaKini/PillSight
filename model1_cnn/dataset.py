import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


def get_train_transform():
    return T.Compose([
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=cfg.ROTATION_DEGREES),
        T.ColorJitter(brightness=cfg.COLOR_JITTER_BRIGHTNESS,
                      contrast=cfg.COLOR_JITTER_CONTRAST),
        T.ToTensor(),
        T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])


def get_eval_transform():
    # no augmentation for val/test
    return T.Compose([
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])


class PillDataset(Dataset):
    def __init__(self, df, class_map, transform=None, image_dir=cfg.IMAGE_DIR):
        self.df        = df.reset_index(drop=True)
        self.class_map = class_map
        self.transform = transform if transform is not None else get_eval_transform()
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = Path(str(row[cfg.COL_IMAGE_PATH]))
        label    = str(row[cfg.COL_LABEL])

        if not img_path.is_absolute() and not img_path.exists():
            img_path = self.image_dir / img_path

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.class_map[label]

    def get_class_weights(self):
        # pandas reads label_code_id back as int64, but class_map keys are strings
        # (JSON forces string keys), so we cast before mapping to avoid silent NaNs
        counts = (
            self.df[cfg.COL_LABEL]
                .astype(str)
                .map(self.class_map)
                .value_counts()
                .sort_index()
        )

        all_counts = torch.zeros(len(self.class_map))
        for idx, count in counts.items():
            all_counts[idx] = count
        all_counts = all_counts.clamp(min=1)

        weights = 1.0 / all_counts
        weights = weights / weights.sum() * len(self.class_map)
        return weights


def build_datasets(
    filtered_csv=cfg.FILTERED_METADATA_CSV,
    class_map_json=cfg.CLASS_MAP_JSON,
):
    df = pd.read_csv(filtered_csv)
    with open(class_map_json) as f:
        class_map = json.load(f)

    # split off test first, then split remaining into train/val
    train_val, test_df = train_test_split(
        df,
        test_size=cfg.TEST_RATIO,
        stratify=df[cfg.COL_LABEL],
        random_state=cfg.RANDOM_SEED,
    )
    val_fraction = cfg.VAL_RATIO / (cfg.TRAIN_RATIO + cfg.VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val[cfg.COL_LABEL],
        random_state=cfg.RANDOM_SEED,
    )

    print("[split] train=%d  val=%d  test=%d" % (len(train_df), len(val_df), len(test_df)))

    train_ds = PillDataset(train_df, class_map, transform=get_train_transform())
    val_ds   = PillDataset(val_df,   class_map, transform=get_eval_transform())
    test_ds  = PillDataset(test_df,  class_map, transform=get_eval_transform())

    return train_ds, val_ds, test_ds
