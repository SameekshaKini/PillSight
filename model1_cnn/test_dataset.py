# quick sanity check -- run this to confirm dataset.py loads correctly
# not part of the final pipeline, just for verification

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_datasets
import config as cfg

print("building datasets...")
train_ds, val_ds, test_ds = build_datasets()

print("train size : %d" % len(train_ds))
print("val size   : %d" % len(val_ds))
print("test size  : %d" % len(test_ds))
print()

# load one batch
loader = DataLoader(train_ds, batch_size=8, shuffle=True)
images, labels = next(iter(loader))

print("batch image shape : %s  (batch, channels, height, width)" % str(tuple(images.shape)))
print("batch label tensor: %s" % labels.tolist())
print("label dtype       : %s" % labels.dtype)
print("image dtype       : %s" % images.dtype)
print("image value range : %.3f to %.3f" % (images.min().item(), images.max().item()))
print()

# map indices back to class names
with open(cfg.CLASS_MAP_JSON) as f:
    class_map = json.load(f)
idx_to_class = {v: k for k, v in class_map.items()}

print("class names for this batch:")
for i, lbl in enumerate(labels.tolist()):
    print("  sample %d -> label_code_id=%s" % (i, idx_to_class[lbl]))

print()
print("dataset.py check passed")
