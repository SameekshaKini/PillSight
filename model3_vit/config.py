"""
model3_vit/config.py
All hyperparameters and paths in one place — mirrors model1_cnn/config.py structure.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
# Works whether you're running locally or from Colab (set COLAB_ROOT env var)
_ROOT = os.environ.get("COLAB_ROOT", os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR          = os.path.join(_ROOT, "data")
CSV_PATH          = os.path.join(DATA_DIR, "all_labels.csv")
FILTERED_CSV      = os.path.join(DATA_DIR, "filtered_metadata.csv")   # written by data_exploration.py
CLASS_MAP_PATH    = os.path.join(DATA_DIR, "class_map.json")           # written by data_exploration.py
IMAGE_ROOT        = os.path.join(DATA_DIR, "classification_data")

CHECKPOINT_DIR    = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), "outputs")
GRADCAM_DIR       = os.path.join(OUTPUT_DIR, "gradcam")

# ── Dataset ────────────────────────────────────────────────────────────────────
NUM_CLASSES   = 10          # top-N manufacturer groups; change here + re-run data_exploration.py
RANDOM_SEED   = 42
VAL_SPLIT     = 0.15        # fraction of training data used for validation
TEST_SPLIT    = 0.15        # fraction of full dataset held out as test

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "vit_b_16"  # torchvision key
IMAGE_SIZE    = 224         # ViT-B/16 expects 224×224
PATCH_SIZE    = 16          # baked into ViT-B/16
PRETRAINED    = True        # start from ImageNet-21k weights

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LR            = 3e-4        # base learning rate for the classifier head
BACKBONE_LR   = 5e-5        # lower LR for the frozen→unfrozen ViT backbone
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1         # label-smoothing epsilon
UNFREEZE_EPOCH = 3          # epoch at which backbone layers are unfrozen

# ── Augmentation ───────────────────────────────────────────────────────────────
MIXUP_ALPHA   = 0.2         # 0 = disabled
COLOR_JITTER  = 0.3
RAND_ERASING  = 0.1

# ── Misc ───────────────────────────────────────────────────────────────────────
NUM_WORKERS   = 2           # DataLoader workers (set 0 on Windows / Colab if issues)
PIN_MEMORY    = True
DEVICE        = "cuda"      # overridden at runtime if CUDA unavailable
