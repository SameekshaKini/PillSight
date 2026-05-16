"""
model3_vit/train.py
Two-phase training with mixup for small dataset ViT fine-tuning.
"""
import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import (
    CHECKPOINT_DIR, OUTPUT_DIR, NUM_EPOCHS,
    HEAD_LR, FINETUNE_LR, WEIGHT_DECAY,
    LABEL_SMOOTH, UNFREEZE_EPOCH, WARMUP_EPOCHS,
    DEVICE, RANDOM_SEED, MIXUP_ALPHA,
)
from dataset import get_dataloaders
from model import build_model

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def mixup(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)


def make_scheduler(optimizer, warmup, total):
    w = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    c = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-8)
    return SequentialLR(optimizer, [w, c], milestones=[warmup])


def run_val(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, labels).item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def run_train(model, loader, criterion, optimizer, device, alpha):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, y_a, y_b, lam = mixup(imgs, labels, alpha)
        logits = model(imgs)
        loss   = mixup_loss(criterion, logits, y_a, y_b, lam)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (lam*(preds==y_a).float() + (1-lam)*(preds==y_b).float()).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


def train():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    train_loader, val_loader, _, _ = get_dataloaders(verbose=True)
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc, best_epoch = 0.0, 0

    # ── Phase 1: head only (NO mixup yet — head needs clean signal) ───────────
    print(f"\n[train] Phase 1: frozen backbone, {UNFREEZE_EPOCH} epochs")
    opt1  = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    sch1  = make_scheduler(opt1, WARMUP_EPOCHS, UNFREEZE_EPOCH)

    for ep in range(1, UNFREEZE_EPOCH + 1):
        t0 = time.time()
        tl, ta = run_train(model, train_loader, criterion, opt1, device, alpha=0.0)
        vl, va = run_val(model, val_loader, criterion, device)
        sch1.step()
        _log(ep, NUM_EPOCHS, tl, ta, vl, va, time.time()-t0)
        _record(history, tl, ta, vl, va)
        if va > best_val_acc:
            best_val_acc, best_epoch = va, ep
            _save(model, CHECKPOINT_DIR)

    # ── Phase 2: full fine-tune WITH mixup ────────────────────────────────────
    remaining = NUM_EPOCHS - UNFREEZE_EPOCH
    print(f"\n[train] Phase 2: full fine-tune, {remaining} epochs, mixup={MIXUP_ALPHA}")
    model.unfreeze_all()
    total, trainable = model.stats()
    print(f"[train] trainable: {trainable:,} / {total:,}")

    opt2  = AdamW(model.param_groups(head_lr=HEAD_LR, backbone_lr=FINETUNE_LR),
                  weight_decay=WEIGHT_DECAY)
    sch2  = make_scheduler(opt2, WARMUP_EPOCHS, remaining)

    for ep in range(UNFREEZE_EPOCH + 1, NUM_EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_train(model, train_loader, criterion, opt2, device, alpha=MIXUP_ALPHA)
        vl, va = run_val(model, val_loader, criterion, device)
        sch2.step()
        _log(ep, NUM_EPOCHS, tl, ta, vl, va, time.time()-t0)
        _record(history, tl, ta, vl, va)
        if va > best_val_acc:
            best_val_acc, best_epoch = va, ep
            _save(model, CHECKPOINT_DIR)

    print(f"\n[train] done — best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    _plot(history, OUTPUT_DIR)
    json.dump(history, open(os.path.join(OUTPUT_DIR, "history.json"), "w"), indent=2)


def _log(ep, total, tl, ta, vl, va, t):
    print(f"Epoch {ep:02d}/{total}  "
          f"train_loss={tl:.4f}  train_acc={ta:.4f}  "
          f"val_loss={vl:.4f}  val_acc={va:.4f}  ({t:.0f}s)")

def _record(h, tl, ta, vl, va):
    h["train_loss"].append(tl); h["train_acc"].append(ta)
    h["val_loss"].append(vl);   h["val_acc"].append(va)

def _save(model, ckpt_dir):
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
    print(f"  ✓ checkpoint saved")

def _plot(history, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"],   label="val")
    ax1.axvline(UNFREEZE_EPOCH, ls="--", color="gray", label="unfreeze")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"],   label="val")
    ax2.axvline(UNFREEZE_EPOCH, ls="--", color="gray", label="unfreeze")
    ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150)
    plt.close()
    print(f"[train] curves saved")


if __name__ == "__main__":
    train()
