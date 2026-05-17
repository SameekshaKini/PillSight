"""
GradCAM for ViT. We hook into the last transformer encoder block's
layer_norm since ViT doesn't have conv layers.
"""
import os, json, random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import *
from model import build_vit

def reshape_transform(tensor, height=14, width=14):
    # tensor shape: [B, 197, 768] → strip cls token → reshape to [B, C, H, W]
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)

test_df = pd.read_csv(f"{DATA_DIR}/test_split_vit.csv")
model   = build_vit(NUM_CLASSES, freeze_backbone=False).to(device)
model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pth", map_location=device))
model.eval()

# Target layer for ViT: last encoder block's LayerNorm
target_layer = [model.encoder.layers[-1].ln_1]

cam = GradCAM(model=model, target_layers=target_layer,
              reshape_transform=reshape_transform)

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

os.makedirs(f"{OUTPUT_DIR}/gradcam", exist_ok=True)
samples = test_df.sample(GRADCAM_SAMPLES, random_state=42)

for i, (_, row) in enumerate(samples.iterrows()):
    img_path = f"{IMAGE_DIR}/{row['image_path']}"
    raw_img  = np.array(Image.open(img_path).convert("RGB").resize((224, 224))) / 255.0
    input_t  = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    pred_class = model(input_t).argmax(1).item()
    targets    = [ClassifierOutputTarget(pred_class)]
    grayscale  = cam(input_tensor=input_t, targets=targets)
    overlay    = show_cam_on_image(raw_img.astype(np.float32), grayscale[0], use_rgb=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(raw_img); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(overlay);  plt.title(f"GradCAM (pred={pred_class})"); plt.axis("off")
    plt.savefig(f"{OUTPUT_DIR}/gradcam/sample_{i}.png", dpi=150)
    plt.close()

print(f"GradCAM images saved to {OUTPUT_DIR}/gradcam/")