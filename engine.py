"""训练、验证、测试共用的一轮循环，以及 SMP 指标计算。"""

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import yaml
from tqdm.auto import tqdm


def read_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for key in ("data_root", "output_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    if config["num_classes"] != 21 or config["ignore_index"] != 255:
        raise ValueError("VOC 2012 应使用 num_classes=21、ignore_index=255。")
    if not 0 < config["val_ratio"] < 1:
        raise ValueError("val_ratio 必须在 0 与 1 之间。")
    if len(config["image_size"]) != 2 or min(config["image_size"]) < 8:
        raise ValueError("image_size 应为 [高度, 宽度]，且两个值都至少为 8。")
    for key in ("batch_size", "epochs", "base_channels", "learning_rate"):
        if config[key] <= 0:
            raise ValueError(f"{key} 必须大于 0。")
    return config


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 PyTorch 安装，或将 device 改为 cpu。")
    return torch.device(name)


def run_epoch(model, loader, criterion, device, config, optimizer=None, desc=""):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    pixel_count = 0
    # 累加所有 batch 的计数，再算整轮指标，避免直接平均 batch 的 mIoU。
    totals = torch.zeros(4, 1, config["num_classes"], dtype=torch.long)

    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc=desc)
        for images, masks in progress:
            images = images.to(device)
            masks = masks.to(device)
            valid_pixels = (masks != config["ignore_index"]).sum().item()
            if valid_pixels == 0:
                continue  # 极端情况下整批都是忽略区域，不计算损失。

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)             # (B, 21, H, W)
            batch_loss_sum = criterion(logits, masks)  # criterion 使用 reduction="sum"
            loss = batch_loss_sum / valid_pixels
            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.detach().argmax(dim=1)  # (B, H, W)，类别编号
            # 在 CPU 统计指标，兼容不同 SMP 版本的 multiclass 实现。
            stats = smp.metrics.get_stats(
                predictions.cpu(), masks.cpu(), mode="multiclass",
                num_classes=config["num_classes"], ignore_index=config["ignore_index"],
            )
            totals += torch.stack([value.sum(dim=0, keepdim=True) for value in stats])
            loss_sum += batch_loss_sum.detach().item()
            pixel_count += valid_pixels
            progress.set_postfix(loss=f"{loss_sum / pixel_count:.4f}")

    if pixel_count == 0:
        raise ValueError("本轮没有有效标签像素，请检查数据集和 mask。")
    tp, fp, fn, tn = totals
    scores = {"loss": loss_sum / pixel_count}
    functions = {
        "miou": smp.metrics.iou_score,
        "f1_score": smp.metrics.f1_score,
        "precision": smp.metrics.precision,
        "recall": smp.metrics.recall,
    }
    for name, function in functions.items():
        scores[name] = function(
            tp, fp, fn, tn, reduction="macro", zero_division=0,
        ).item()
    # 多类别像素准确率：正确像素 / 有效像素；不要用 one-vs-rest 的 accuracy 替代。
    scores["pixel_accuracy"] = (tp.sum().float() / (tp + fn).sum()).item()
    return scores, totals
