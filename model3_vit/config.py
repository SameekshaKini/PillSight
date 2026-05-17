import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR        = os.path.join(BASE_DIR, "data")
IMAGE_DIR       = os.path.join(DATA_DIR, "classification_data", "segmented_nih_pills_224")
CSV_PATH        = os.path.join(DATA_DIR, "all_labels.csv")
FILTERED_CSV    = os.path.join(DATA_DIR, "filtered_metadata.csv")   # reuse from model1
CLASS_MAP_PATH  = os.path.join(DATA_DIR, "class_map.json")          # reuse from model1
CHECKPOINT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "outputs")

# ── Dataset ────────────────────────────────────────────────────────────────
NUM_CLASSES     = 10
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
# TEST_SPLIT is the remainder (0.15)
RANDOM_SEED     = 42

# ── ViT model ──────────────────────────────────────────────────────────────
MODEL_NAME      = "vit_b_16"   # torchvision identifier
IMAGE_SIZE      = 224          # ViT-B/16 native resolution

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_EPOCHS      = 20
LEARNING_RATE   = 2e-5         # small LR — we're fine-tuning a pretrained ViT
WEIGHT_DECAY    = 1e-4
UNFREEZE_EPOCH  = 5            # epoch at which backbone unfreezes (2-phase training)

# ── GradCAM ────────────────────────────────────────────────────────────────
GRADCAM_SAMPLES = 5