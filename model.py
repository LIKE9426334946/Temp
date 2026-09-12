"""两次下采样的小型 U-Net，从零训练，不下载预训练权重。"""

import torch
from torch import nn
from torch.nn import functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    def __init__(self, num_classes=21, base_channels=32):
        super().__init__()
        c = base_channels
        self.encoder1 = double_conv(3, c)
        self.encoder2 = double_conv(c, c * 2)
        self.bottleneck = double_conv(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.decoder2 = double_conv(c * 4 + c * 2, c * 2)
        self.decoder1 = double_conv(c * 2 + c, c)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)                  # (B, c, H, W)
        x2 = self.encoder2(self.pool(x1))      # (B, 2c, H/2, W/2)
        x3 = self.bottleneck(self.pool(x2))    # (B, 4c, H/4, W/4)

        x = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder2(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder1(torch.cat([x, x1], dim=1))
        # 返回原始 logits；CrossEntropyLoss 内部会处理 softmax。
        return self.head(x)                   # (B, 21, H, W)
