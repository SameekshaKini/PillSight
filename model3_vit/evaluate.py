import os, json
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, top_k_accuracy_score)
import seaborn as sns

from config import *
from dataset import PillDataset
from model import build_vit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)
idx_to_class = {v: k for k, v in class_map.items()}

test_df   = pd.read_csv(f"{DATA_DIR}/test_split_vit.csv")
test_ds   = PillDataset(test_df, "test")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = build_vit(NUM_CLASSES, freeze_backbone=False).to(device)
model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pth", map_location=device))
model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(lbls.numpy())

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

top1 = (all_preds == all_labels).mean()
top3 = top_k_accuracy_score(all_labels, all_probs, k=3)
f1   = f1_score(all_labels, all_preds, average="macro")

print(f"Top-1 Accuracy : {top1:.4f}")
print(f"Top-3 Accuracy : {top3:.4f}")
print(f"Macro F1       : {f1:.4f}")
print("\n" + classification_report(all_labels, all_preds))

# Confusion matrix
os.makedirs(OUTPUT_DIR, exist_ok=True)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("ViT-B/16 Confusion Matrix")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
print(f"Confusion matrix saved to {OUTPUT_DIR}/")