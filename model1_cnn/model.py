import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


class ConvBlock(nn.Module):
    # one conv block: Conv -> BatchNorm -> ReLU -> MaxPool
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class PillCNN(nn.Module):
    def __init__(self, num_classes=cfg.NUM_CLASSES, dropout_rate=cfg.DROPOUT_RATE):
        super().__init__()

        # 3 conv blocks, each halves spatial size and doubles channels
        self.conv1 = ConvBlock(3,   32)   # 224 -> 112
        self.conv2 = ConvBlock(32,  64)   # 112 -> 56
        self.conv3 = ConvBlock(64, 128)   #  56 -> 28

        # adaptive pool so the FC head doesn't break if we change IMAGE_SIZE
        self.pool = nn.AdaptiveAvgPool2d((4, 4))   # 28 -> 4x4, gives 128*4*4 = 2048 features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x  # raw logits, CrossEntropyLoss handles softmax

    def get_gradcam_target_layer(self):
        return self.conv3.block[0]  # last conv layer before the FC head


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PillCNN(num_classes=cfg.NUM_CLASSES)
    out   = model(torch.zeros(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
    print("output shape:", out.shape)
    print("parameters  :", count_parameters(model))
