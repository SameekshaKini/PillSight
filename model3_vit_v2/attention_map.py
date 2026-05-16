"""
model3_vit/attention_map.py
Attention Rollout visualization for ViT-B/16.

Uses forward-patching (not hooks) to avoid recursion.
"""
import os, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import CHECKPOINT_DIR, OUTPUT_DIR, IMAGE_SIZE, IMAGE_ROOT, FILTERED_CSV, CLASS_MAP_PATH, DEVICE
from model import build_model

ATTN_DIR    = os.path.join(OUTPUT_DIR, "attention_maps")
NUM_SAMPLES = 5
os.makedirs(ATTN_DIR, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def extract_attentions(model, tensor):
    """
    Patch each block's self_attention.forward to capture attention weights.
    No hooks = no recursion.
    """
    captured = []
    original = {}

    for i, block in enumerate(model.vit.encoder.layers):
        original[i] = block.self_attention.forward

        def make_patched(orig):
            def patched(q, k, v, **kwargs):
                kwargs["need_weights"]        = True
                kwargs["average_attn_weights"] = False
                out, attn = orig(q, k, v, **kwargs)
                captured.append(attn.detach().cpu())
                return out, attn
            return patched

        block.self_attention.forward = make_patched(block.self_attention.forward)

    with torch.no_grad():
        logits = model(tensor)

    for i, block in enumerate(model.vit.encoder.layers):
        block.self_attention.forward = original[i]

    return logits, captured


def rollout(attn_list):
    """
    attn_list: list of [1, heads, seq, seq] tensors
    Returns [seq-1] numpy array (CLS → patches attention)
    """
    seq = attn_list[0].size(-1)
    R   = torch.eye(seq)

    for attn in attn_list:
        # [1, heads, seq, seq] → [seq, seq]
        A = attn.squeeze(0).mean(0)
        A = A + torch.eye(seq)
        A = A / A.sum(-1, keepdim=True)
        R = A @ R

    mask = R[0, 1:]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask.numpy()


def visualize():
    import pandas as pd
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = build_model(device)
    model.unfreeze_all()
    ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with open(CLASS_MAP_PATH) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    df      = pd.read_csv(FILTERED_CSV)
    samples = df.sample(NUM_SAMPLES, random_state=42)

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    saved = 0
    for _, row in samples.iterrows():
        img_path = os.path.join(IMAGE_ROOT, row["image_path"])
        if not os.path.exists(img_path):
            print(f"[attn] skipping missing: {img_path}")
            continue

        pil   = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        inp   = tf(pil).unsqueeze(0).to(device)

        logits, attns = extract_attentions(model, inp)

        pred      = logits.argmax(1).item()
        pred_name = idx_to_class[pred]
        true_name = idx_to_class.get(class_to_idx.get(str(row["label_code_id"])), "?")

        mask      = rollout(attns)
        grid      = int(math.sqrt(len(mask)))
        mask_2d   = mask.reshape(grid, grid)
        mask_up   = np.array(Image.fromarray(
            (mask_2d * 255).astype(np.uint8)
        ).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out  = os.path.join(ATTN_DIR, f"{stem}_attn.png")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(pil);              axes[0].set_title("Original");     axes[0].axis("off")
        axes[1].imshow(mask_up, cmap="inferno"); axes[1].set_title("Attention map"); axes[1].axis("off")
        axes[2].imshow(pil); axes[2].imshow(mask_up, cmap="inferno", alpha=0.5)
        axes[2].set_title(f"Pred: {pred_name}\nTrue: {true_name}", fontsize=8)
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=150); plt.close()
        saved += 1

    print(f"[attention_map] saved {saved} maps → {ATTN_DIR}")


if __name__ == "__main__":
    visualize()
