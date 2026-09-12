"""读取 VOC 2012：图片缩放到 [0, 1]，mask 保留整数类别编号。"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as TF


CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


class VOCTransform:
    def __init__(self, image_size, augment=False):
        self.height, self.width = image_size
        self.augment = augment

    def __call__(self, image, mask):
        size = (self.width, self.height)  # PIL 的顺序是 (宽度, 高度)
        image = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        # 必须使用最近邻插值，不能把类别编号插值成小数。
        mask = mask.resize(size, Image.Resampling.NEAREST)

        # 图片与标签必须一起翻转，验证和测试时不随机翻转。
        if self.augment and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)  # float32: (3, H, W)，范围 [0, 1]
        # 不对 mask 使用 ToTensor，不除以 255，也不转成 RGB。
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # (H, W)
        return image, mask


def make_dataset(config, split, augment=False, download=None):
    return VOCSegmentation(
        root=config["data_root"],
        year="2012",
        image_set=split,
        download=config["download"] if download is None else download,
        transforms=VOCTransform(config["image_size"], augment=augment),
    )


def make_train_val(config):
    train_data = make_dataset(config, "train", augment=True)
    # 单独创建一份带确定性预处理的数据集，底层仍读取相同文件。
    val_data = make_dataset(config, "train", download=False)
    generator = torch.Generator().manual_seed(config["seed"])
    indices = torch.randperm(len(train_data), generator=generator).tolist()
    val_count = max(1, int(len(indices) * config["val_ratio"]))
    if val_count >= len(indices):
        raise ValueError("数据太少，无法同时划分训练集和验证集。")
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return Subset(train_data, train_indices), Subset(val_data, val_indices)
