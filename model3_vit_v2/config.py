"""
model3_vit/config.py
Tuned for small dataset (~2300 images, 10 classes) — heavy regularization.
"""
import os

_ROOT = os.environ.get("COLAB_ROOT", os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR       = os.path.join(_ROOT, "data")
CSV_PATH       = os.path.join(DATA_DIR, "all_labels.csv")
FILTERED_CSV   = os.path.join(DATA_DIR, "filtered_metadata.csv")
CLASS_MAP_PATH = os.path.join(DATA_DIR, "class_map.json")
IMAGE_ROOT     = os.path.join(DATA_DIR, "classification_data")

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Dataset
NUM_CLASSES  = 10
RANDOM_SEED  = 42
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15

# Model
IMAGE_SIZE   = 224
PRETRAINED   = True

# Training
BATCH_SIZE      = 32    # smaller batch = more gradient noise = less overfit
NUM_EPOCHS      = 50
HEAD_LR         = 5e-4  # head trains faster during frozen phase
FINETUNE_LR     = 5e-6  # very conservative backbone LR — 4x lower than before
WEIGHT_DECAY    = 0.05  # strong L2 — standard for ViT fine-tuning
LABEL_SMOOTH    = 0.2   # higher smoothing = less confident = less overfit
UNFREEZE_EPOCH  = 15    # keep frozen longer — let head really stabilize
WARMUP_EPOCHS   = 3

# Augmentation — aggressive to fight overfitting
COLOR_JITTER    = 0.4
RANDOM_ERASING  = 0.4   # erase up to 40% of patches randomly
MIXUP_ALPHA     = 0.4   # re-enable mixup — helps ViT generalize

NUM_WORKERS  = 4
PIN_MEMORY   = True
DEVICE       = "cuda"
