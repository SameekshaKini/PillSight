# train.py
# run data_exploration.py first to generate filtered_metadata.csv and class_map.json

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from dataset import build_datasets
from model import PillCNN, count_parameters


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using Apple MPS")
    else:
        device = torch.device("cpu")
        print("using CPU (this will be slow)")
    return device


def build_loaders(train_ds, val_ds):
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, device, class_weights):
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    # halve LR if val accuracy stops improving for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print("\n%6s  %10s  %9s  %9s  %8s  %6s" % ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time"))
    print("-" * 58)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = "  <-- best" if val_acc > best_val_acc else ""
        print("%6d  %10.4f  %9.4f  %9.4f  %8.4f  %5.1fs%s" % (
            epoch, train_loss, train_acc, val_loss, val_acc, time.time() - t0, marker
        ))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": cfg.NUM_CLASSES,
            }, cfg.BEST_MODEL_PATH)

    print("\nbest val acc: %.4f  saved to %s" % (best_val_acc, cfg.BEST_MODEL_PATH))
    return history


def save_learning_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"],   label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"],   label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("accuracy")
    ax2.legend()

    plt.tight_layout()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.OUTPUT_DIR / "learning_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("learning curves saved to %s" % out)


def main():
    device = get_device()

    train_ds, val_ds, test_ds = build_datasets()
    train_loader, val_loader  = build_loaders(train_ds, val_ds)

    class_weights = train_ds.get_class_weights()
    print("class weights:", [round(w, 4) for w in class_weights.tolist()])

    model = PillCNN(num_classes=cfg.NUM_CLASSES, dropout_rate=cfg.DROPOUT_RATE).to(device)
    print("parameters:", count_parameters(model))

    history = train(model, train_loader, val_loader, device, class_weights)
    save_learning_curves(history)
    print("\ndone -- run evaluate.py next")


if __name__ == "__main__":
    main()
