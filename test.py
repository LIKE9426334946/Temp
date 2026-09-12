"""用官方 val 做最终本地测试：python test.py --config config.yaml"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示器、服务器和 Notebook 中也可以保存图片。
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, make_dataset
from engine import get_device, read_config, run_epoch
from model import SmallUNet


def voc_palette():
    """VOC 类别调色板；第 255 类（忽略区域）用白色显示。"""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for index in range(256):
        value = index
        for bit in range(8):
            for channel in range(3):
                palette[index, channel] |= ((value >> channel) & 1) << (7 - bit)
            value >>= 3
    palette[255] = [255, 255, 255]
    return palette


@torch.no_grad()
def save_predictions(model, dataset, device, output_dir, count):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = voc_palette()
    for index in range(min(count, len(dataset))):
        image, mask = dataset[index]
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
        name = Path(dataset.images[index]).stem
        # 原始预测：每个像素是 0~20 的类别编号，方便继续读取分析。
        Image.fromarray(prediction.astype(np.uint8)).save(output_dir / f"{name}_mask.png")

        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        panels = [image.permute(1, 2, 0).numpy(), palette[mask.numpy()], palette[prediction]]
        for axis, panel, title in zip(axes, panels, ["Image", "Ground truth", "Prediction"]):
            axis.imshow(panel)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"VOC 2012 / {name} | White in ground truth = ignored")
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}_comparison.png", dpi=140)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config = read_config(args.config)
    device = get_device(config["device"])
    output_dir = Path(config["output_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 {checkpoint_path}，请先运行 train.py。")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # 模型结构和输入大小跟随训练记录；数据路径、设备等使用当前配置。
    for key in ("num_classes", "base_channels", "ignore_index", "image_size"):
        config[key] = checkpoint["config"][key]
    model = SmallUNet(config["num_classes"], config["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = make_dataset(config, "val")
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config["ignore_index"], reduction="sum")
    print(f"Checkpoint epoch: {checkpoint['epoch']} | Test images: {len(dataset)} | Device: {device}")
    scores, totals = run_epoch(model, loader, criterion, device, config, desc="Test")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "split": "VOC2012 official val, held out as local test",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "num_images": len(dataset),
        "image_size": config["image_size"],
        "averaging": "macro over 21 classes including background; zero_division=0",
        **scores,
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps(scores, indent=2))

    per_class_iou = smp.metrics.iou_score(*totals, reduction="none", zero_division=0)[0]
    per_class_f1 = smp.metrics.f1_score(*totals, reduction="none", zero_division=0)[0]
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "class_name", "iou", "f1_score"])
        for index, name in enumerate(CLASS_NAMES):
            writer.writerow([index, name, per_class_iou[index].item(), per_class_f1[index].item()])
    save_predictions(model, dataset, device, output_dir / "predictions", config["num_visualizations"])
    print(f"Test complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
