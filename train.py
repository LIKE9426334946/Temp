"""运行：python train.py --config config.yaml"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dataset import make_train_val
from engine import get_device, read_config, run_epoch
from model import SmallUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = read_config(args.config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = get_device(config["device"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    train_data, val_data = make_train_val(config)
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_data, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )
    # 保存本次划分的图片名，方便核对、复现实验。
    splits = {
        "train": [Path(train_data.dataset.images[i]).stem for i in train_data.indices],
        "validation": [Path(val_data.dataset.images[i]).stem for i in val_data.indices],
        "test": "VOC2012 official val (only used by test.py)",
    }
    with (output_dir / "split.json").open("w", encoding="utf-8") as file:
        json.dump(splits, file, indent=2)

    model = SmallUNet(config["num_classes"], config["base_channels"]).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config["ignore_index"], reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_miou = -1.0
    print(f"Device: {device} | Train: {len(train_data)} | Validation: {len(val_data)}")

    # 每次重新训练都会覆盖同一 output_dir 中的训练记录和最佳模型。
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for epoch in range(1, config["epochs"] + 1):
            train_scores, _ = run_epoch(
                model, train_loader, criterion, device, config,
                optimizer=optimizer, desc=f"Epoch {epoch}/{config['epochs']} train",
            )
            val_scores, _ = run_epoch(
                model, val_loader, criterion, device, config, desc="Validation",
            )
            row = {"epoch": epoch}
            row.update({f"train_{key}": value for key, value in train_scores.items()})
            row.update({f"val_{key}": value for key, value in val_scores.items()})
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            file.flush()

            print(
                f"Epoch {epoch}: train_loss={train_scores['loss']:.4f}, "
                f"val_loss={val_scores['loss']:.4f}, val_mIoU={val_scores['miou']:.4f}, "
                f"val_F1={val_scores['f1_score']:.4f}"
            )
            if val_scores["miou"] > best_miou:
                best_miou = val_scores["miou"]
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_miou": best_miou,
                }, output_dir / "best_model.pth")
                print(f"Saved best_model.pth (validation mIoU={best_miou:.4f})")

    print(f"Training complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
