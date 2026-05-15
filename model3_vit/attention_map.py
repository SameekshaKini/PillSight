"""
model3_vit/attention_map.py
Visualize ViT attention maps using Attention Rollout.

Unlike CNNs, ViT doesn't use Grad-CAM — instead we roll out the attention weights
across all transformer layers to produce a saliency map showing which image patches
the model attended to when making its prediction.

Saves overlays to outputs/attention_maps/.

Run from inside model3_vit/:
    python attention_map.py
"""

import os, json, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import (
    CHECKPOINT_DIR, OUTPUT_DIR, IMAGE_SIZE,
    IMAGE_ROOT, FILTERED_CSV, CLASS_MAP_PATH, DEVICE,
)
from model import build_model

ATTN_DIR   = os.path.join(OUTPUT_DIR, "attention_maps")
NUM_SAMPLES = 5
os.makedirs(ATTN_DIR, exist_ok=True)


# ── Attention Rollout ──────────────────────────────────────────────────────────

def attention_rollout(attn_weights: list[torch.Tensor]) -> np.ndarray:
    """
    attn_weights: list of [heads, seq_len, seq_len] tensors (one per layer).
    Returns a (seq_len-1,) array of attention values for the [CLS] token.
    """
    rollout = torch.eye(attn_weights[0].size(-1), device=attn_weights[0].device)
    for attn in attn_weights:
        attn_mean = attn.mean(dim=0)           # average over heads
        attn_mean = attn_mean + torch.eye(attn_mean.size(-1), device=attn_mean.device)
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
        rollout = attn_mean @ rollout

    # CLS token (index 0) attention over patch tokens (index 1:)
    mask = rollout[0, 1:]
    mask = mask / mask.max()
    return mask.cpu().numpy()


def register_hooks(model):
    """Attach forward hooks to every attention block; returns list that fills in-place."""
    attentions = []

    def hook_fn(module, input, output):
        # torchvision ViT: EncoderBlock.self_attention is nn.MultiheadAttention
        # output is (attn_output, attn_weights); need_weights must be True
        attentions.append(output[1].detach())

    handles = []
    for block in model.vit.encoder.layers:
        h = block.self_attention.register_forward_hook(hook_fn)
        handles.append(h)

    return attentions, handles


# ── Main ───────────────────────────────────────────────────────────────────────

def visualize():
    import pandas as pd

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = build_model(device)
    model.unfreeze_backbone()
    ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with open(CLASS_MAP_PATH) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    df = pd.read_csv(FILTERED_CSV)
    samples = df.sample(NUM_SAMPLES, random_state=42)

    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    for _, row in samples.iterrows():
        img_path = os.path.join(IMAGE_ROOT, row["image_path"])
        pil_img  = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        tensor   = preprocess(pil_img).unsqueeze(0).to(device)

        attentions, handles = register_hooks(model)
        with torch.no_grad():
            logits = model(tensor)
        for h in handles:
            h.remove()

        pred_idx  = logits.argmax(1).item()
        pred_name = idx_to_class[pred_idx]
        true_name = idx_to_class.get(class_to_idx.get(str(row["label_code_id"])), "?")

        mask = attention_rollout(attentions)
        grid_size = int(math.sqrt(len(mask)))   # 14 for ViT-B/16 on 224px
        mask = mask.reshape(grid_size, grid_size)
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE)))

        _save_overlay(pil_img, mask, pred_name, true_name, img_path)

    print(f"[attention_map] saved {NUM_SAMPLES} maps to {ATTN_DIR}")


def _save_overlay(pil_img, mask, pred, true_label, img_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(pil_img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(mask, cmap="inferno"); axes[1].set_title("Attention map"); axes[1].axis("off")
    axes[2].imshow(pil_img)
    axes[2].imshow(mask, cmap="inferno", alpha=0.5)
    axes[2].set_title(f"Overlay\nPred: {pred}\nTrue: {true_label}", fontsize=8)
    axes[2].axis("off")

    plt.tight_layout()
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out  = os.path.join(ATTN_DIR, f"{stem}_attention.png")
    plt.savefig(out, dpi=150)
    plt.close()


if __name__ == "__main__":
    visualize()
