"""
model3_vit/train.py
Training loop for ViT-B/16 pill classifier.

Run from inside model3_vit/:
    python train.py
"""

import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    CHECKPOINT_DIR, OUTPUT_DIR,
    NUM_EPOCHS, LR, BACKBONE_LR, WEIGHT_DECAY, LABEL_SMOOTH,
    UNFREEZE_EPOCH, MIXUP_ALPHA, DEVICE, RANDOM_SEED,
)
from dataset import get_dataloaders
from model import build_model

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Mixup ──────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Training loop ──────────────────────────────────────────────────────────────

def train():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    train_loader, val_loader, _, class_to_idx = get_dataloaders()
    model = build_model(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # Two parameter groups: head gets LR, backbone gets BACKBONE_LR (after unfreeze)
    def get_optimizer(backbone_frozen: bool):
        if backbone_frozen:
            return AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR, weight_decay=WEIGHT_DECAY
            )
        else:
            return AdamW([
                {"params": model.vit.heads.parameters(), "lr": LR},
                {"params": [p for n, p in model.vit.named_parameters()
                            if "heads" not in n],       "lr": BACKBONE_LR},
            ], weight_decay=WEIGHT_DECAY)

    optimizer = get_optimizer(backbone_frozen=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    backbone_frozen = True

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ── Unfreeze backbone at UNFREEZE_EPOCH ────────────────────────────────
        if epoch == UNFREEZE_EPOCH and backbone_frozen:
            print(f"[train] epoch {epoch}: unfreezing ViT backbone")
            model.unfreeze_backbone()
            backbone_frozen = False
            optimizer = get_optimizer(backbone_frozen=False)
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - epoch + 1, eta_min=1e-7)

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA, device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.0f}s)")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ saved best checkpoint  (val_acc={val_acc:.4f})")

    # ── Save learning curves ───────────────────────────────────────────────────
    _plot_history(history, OUTPUT_DIR)
    json.dump(history, open(os.path.join(OUTPUT_DIR, "history.json"), "w"), indent=2)
    print(f"\n[train] done. best val_acc={best_val_acc:.4f}")


def _plot_history(history, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"],   label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"],   label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150)
    plt.close()
    print(f"[train] learning curves saved to {out_dir}/learning_curves.png")


if __name__ == "__main__":
    train()
