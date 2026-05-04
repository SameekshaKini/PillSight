# gradcam.py
# generates grad-cam heatmaps for a few test images to see what the model is looking at
# requires: pip install grad-cam

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from dataset import build_datasets
from model import PillCNN

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError("install pytorch-grad-cam first:  pip install grad-cam")


def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError("no checkpoint at %s -- run train.py first" % checkpoint_path)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = PillCNN(num_classes=ckpt.get("num_classes", cfg.NUM_CLASSES))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def tensor_to_rgb(tensor):
    # undo imagenet normalization so we can display the original image
    mean = torch.tensor(cfg.IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(cfg.IMAGENET_STD).view(3, 1, 1)
    img  = (tensor * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def pick_samples(test_ds, n):
    # try to get one image per class for variety
    with open(cfg.CLASS_MAP_JSON) as f:
        class_map = json.load(f)
    idx_to_class = {v: k for k, v in class_map.items()}

    class_to_idx = {}
    for i in range(len(test_ds)):
        _, label = test_ds[i]
        if label not in class_to_idx:
            class_to_idx[label] = i

    selected = list(class_to_idx.values())[:n]

    # fill up if we somehow don't have enough classes
    if len(selected) < n:
        extra = [i for i in range(len(test_ds)) if i not in selected]
        selected += extra[:n - len(selected)]

    samples = []
    for idx in selected[:n]:
        tensor, label = test_ds[idx]
        samples.append((tensor, label, idx_to_class[label]))
    return samples


def run_gradcam(model, samples, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.CLASS_MAP_JSON) as f:
        class_map = json.load(f)
    idx_to_class = {v: k for k, v in class_map.items()}

    cam = GradCAM(model=model, target_layers=[model.get_gradcam_target_layer()])

    for i, (tensor, true_idx, true_name) in enumerate(samples):
        inp = tensor.unsqueeze(0).to(device)

        # targets=None means grad-cam runs w.r.t. the top predicted class
        heatmap = cam(input_tensor=inp, targets=None)[0]

        with torch.no_grad():
            pred_idx = model(inp).argmax(dim=1).item()
        pred_name = idx_to_class.get(pred_idx, str(pred_idx))

        rgb_img = tensor_to_rgb(tensor)
        cam_img = show_cam_on_image(rgb_img, heatmap, use_rgb=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        ax1.imshow(rgb_img)
        ax1.set_title("original\ntrue: %s" % true_name, fontsize=9)
        ax1.axis("off")

        correct = "correct" if pred_name == true_name else "wrong"
        ax2.imshow(cam_img)
        ax2.set_title("grad-cam (%s)\npred: %s" % (correct, pred_name), fontsize=9)
        ax2.axis("off")

        fig.suptitle("PillSight CNN -- grad-cam", fontsize=11)
        plt.tight_layout()

        out_path = out_dir / ("gradcam_%02d.png" % (i + 1))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print("  [%d/%d] true=%-10s  pred=%-10s  -> %s" % (
            i + 1, len(samples), true_name, pred_name, out_path.name
        ))

    print("\ndone -- saved to %s" % out_dir)


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(cfg.BEST_MODEL_PATH, device)
    _, _, test_ds = build_datasets()
    samples = pick_samples(test_ds, n=cfg.NUM_GRADCAM_SAMPLES)

    print("generating grad-cam for %d images..." % len(samples))
    run_gradcam(model, samples, device, cfg.GRADCAM_DIR)


if __name__ == "__main__":
    main()
