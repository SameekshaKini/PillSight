"""
model3_vit/evaluate.py
Run from inside model3_vit/:  python evaluate.py
"""
import os, json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, top_k_accuracy_score

from config import CHECKPOINT_DIR, OUTPUT_DIR, NUM_CLASSES, DEVICE
from dataset import get_dataloaders
from model import build_model

os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device: {device}")

    _, _, test_loader, class_to_idx = get_dataloaders(verbose=False)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(device)
    model.unfreeze_all()
    ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"[evaluate] loaded: {ckpt}")

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs.to(device))
            probs  = torch.softmax(logits, dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    top1 = (all_labels == all_preds).mean()
    top3 = top_k_accuracy_score(all_labels, all_probs, k=min(3, NUM_CLASSES))
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\n  Top-1 accuracy : {top1:.4f}")
    print(f"  Top-3 accuracy : {top3:.4f}")
    print(f"  Macro F1       : {f1:.4f}")
    print("\n" + classification_report(
        all_labels, all_preds,
        target_names=[idx_to_class[i] for i in range(NUM_CLASSES)],
        zero_division=0
    ))

    json.dump({"top1": top1, "top3": top3, "macro_f1": f1},
              open(os.path.join(OUTPUT_DIR, "metrics.json"), "w"), indent=2)

    _plot_cm(all_labels, all_preds, idx_to_class)


def _plot_cm(labels, preds, idx_to_class):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = range(len(idx_to_class))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([idx_to_class[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([idx_to_class[i] for i in ticks], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("ViT-B/16 Confusion Matrix")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[evaluate] confusion matrix → {out}")


if __name__ == "__main__":
    evaluate()
