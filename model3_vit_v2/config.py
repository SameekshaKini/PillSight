"""
model3_vit/config.py
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

# Training — conservative, proven settings
BATCH_SIZE      = 64     # bigger batch = more stable gradients on A100
NUM_EPOCHS      = 40
HEAD_LR         = 2e-4   # head learning rate (frozen phase)
FINETUNE_LR     = 2e-5   # full model LR after unfreezing
WEIGHT_DECAY    = 1e-2   # AdamW default, strong regularizer
LABEL_SMOOTH    = 0.1
UNFREEZE_EPOCH  = 8      # freeze for 8 epochs, then full fine-tune
WARMUP_EPOCHS   = 2      # linear LR warmup

# Augmentation
COLOR_JITTER    = 0.3
RANDOM_ERASING  = 0.25

NUM_WORKERS  = 4
PIN_MEMORY   = True
DEVICE       = "cuda"
