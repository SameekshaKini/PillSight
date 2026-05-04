# evaluate.py
# loads the best checkpoint and runs it on the held-out test set
# prints top-1, top-3, macro F1, and saves a confusion matrix

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from dataset import build_datasets
from model import PillCNN


def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError("no checkpoint found at %s -- run train.py first" % checkpoint_path)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = PillCNN(num_classes=ckpt.get("num_classes", cfg.NUM_CLASSES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print("loaded checkpoint from epoch %d (val_acc=%.4f)" % (ckpt["epoch"], ckpt["val_acc"]))
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs)


def top_k_accuracy(labels, probs, k):
    top_k   = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(labels[i] in top_k[i] for i in range(len(labels)))
    return correct / len(labels)


def compute_metrics(labels, preds, probs, class_names):
    top1 = (labels == preds).mean()
    top3 = top_k_accuracy(labels, probs, k=3)
    f1   = f1_score(labels, preds, average="macro", zero_division=0)

    print("\n--- results ---")
    print("top-1 accuracy : %.4f  (%.2f%%)" % (top1, top1 * 100))
    print("top-3 accuracy : %.4f  (%.2f%%)" % (top3, top3 * 100))
    print("macro F1       : %.4f" % f1)
    print()
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    return {"top1": float(top1), "top3": float(top3), "macro_f1": float(f1)}


def plot_confusion_matrix(labels, preds, class_names, out_path):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    n        = len(class_names)
    fig_size = max(8, n * 0.8)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("confusion matrix (row-normalized) -- PillSight CNN")

    # annotate each cell with the raw count
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("confusion matrix saved to %s" % out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg.CLASS_MAP_JSON) as f:
        class_map = json.load(f)
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]

    model = load_model(cfg.BEST_MODEL_PATH, device)

    _, _, test_ds = build_datasets()
    test_loader   = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    labels, preds, probs = run_inference(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs, class_names)
    plot_confusion_matrix(labels, preds, class_names, cfg.CONFUSION_MATRIX_PATH)

    # save metrics to json so we can compare against the other models later
    metrics_path = cfg.OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("metrics saved to %s" % metrics_path)


if __name__ == "__main__":
    main()
