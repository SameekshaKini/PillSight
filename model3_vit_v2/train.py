"""
model3_vit/train.py
Two-phase ViT training:
  Phase 1: frozen backbone, train head only
  Phase 2: full fine-tune at lower LR with cosine decay

Run from inside model3_vit/:
    python train.py
"""
import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingLR, SequentialLR
)

from config import (
    CHECKPOINT_DIR, OUTPUT_DIR,
    NUM_EPOCHS, HEAD_LR, FINETUNE_LR, WEIGHT_DECAY,
    LABEL_SMOOTH, UNFREEZE_EPOCH, WARMUP_EPOCHS,
    DEVICE, RANDOM_SEED, NUM_CLASSES,
)
from dataset import get_dataloaders
from model import build_model

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs,
                                eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_acc  += accuracy(logits, labels) * bs
            n          += bs

    return total_loss / n, total_acc / n


def train():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    train_loader, val_loader, _, class_to_idx = get_dataloaders(verbose=True)
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # ── Phase 1: head only ────────────────────────────────────────────────────
    print(f"\n[train] Phase 1: head-only training for {UNFREEZE_EPOCH} epochs")
    optimizer1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler1 = make_scheduler(optimizer1, WARMUP_EPOCHS, UNFREEZE_EPOCH)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, 0

    for epoch in range(1, UNFREEZE_EPOCH + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer1, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer1, device, train=False)
        scheduler1.step()

        _log(epoch, NUM_EPOCHS, tr_loss, tr_acc, vl_loss, vl_acc, time.time() - t0)
        _record(history, tr_loss, tr_acc, vl_loss, vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc, best_epoch = vl_acc, epoch
            _save(model, CHECKPOINT_DIR)

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    remaining = NUM_EPOCHS - UNFREEZE_EPOCH
    print(f"\n[train] Phase 2: full fine-tune for {remaining} epochs (LR={FINETUNE_LR})")
    model.unfreeze_all()
    total, trainable = model.stats()
    print(f"[train] now trainable: {trainable:,} / {total:,}")

    optimizer2 = AdamW(
        model.param_groups(head_lr=HEAD_LR, backbone_lr=FINETUNE_LR),
        weight_decay=WEIGHT_DECAY
    )
    scheduler2 = make_scheduler(optimizer2, WARMUP_EPOCHS, remaining)

    for epoch in range(UNFREEZE_EPOCH + 1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer2, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer2, device, train=False)
        scheduler2.step()

        _log(epoch, NUM_EPOCHS, tr_loss, tr_acc, vl_loss, vl_acc, time.time() - t0)
        _record(history, tr_loss, tr_acc, vl_loss, vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc, best_epoch = vl_acc, epoch
            _save(model, CHECKPOINT_DIR)

    print(f"\n[train] done — best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    _plot(history, OUTPUT_DIR)
    json.dump(history, open(os.path.join(OUTPUT_DIR, "history.json"), "w"), indent=2)


def _log(ep, total, tl, ta, vl, va, elapsed):
    print(f"Epoch {ep:02d}/{total}  "
          f"train_loss={tl:.4f}  train_acc={ta:.4f}  "
          f"val_loss={vl:.4f}  val_acc={va:.4f}  ({elapsed:.0f}s)")


def _record(h, tl, ta, vl, va):
    h["train_loss"].append(tl); h["train_acc"].append(ta)
    h["val_loss"].append(vl);   h["val_acc"].append(va)


def _save(model, ckpt_dir):
    path = os.path.join(ckpt_dir, "best_model.pth")
    torch.save(model.state_dict(), path)
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
    out = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[train] curves saved → {out}")


if __name__ == "__main__":
    train()
