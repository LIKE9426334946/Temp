# VOC 2012 图像分割入门

使用 `torchvision.datasets.VOCSegmentation(year="2012")` 读取数据，训练一个小型 U-Net，为每个像素预测类别。模型由普通卷积、池化、上采样和跳跃连接组成，从零训练，不需要预训练权重。

## 1. 文件说明

| 文件 | 用途 |
| --- | --- |
| `config.yaml` | 数据路径、图片大小、batch size、学习率、训练轮数等 |
| `dataset.py` | 读取 VOC、同步处理图片与标签、划分训练与验证集 |
| `model.py` | 简单的两级 U-Net |
| `engine.py` | 共用训练/评估循环，调用 SMP 统计指标 |
| `train.py` | 训练、验证，保存最佳模型和每轮指标 |
| `test.py` | 加载最佳模型，测试并保存预测对比图 |
| `requirements.txt` | Python 依赖 |

## 2. 安装

建议使用 Python 3.10 或以上版本，并进入本项目目录后运行：

```bash
python -m pip install -r requirements.txt
```

`torch` 和 `torchvision` 必须使用匹配版本。需要 NVIDIA GPU 时，按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 安装适合自己环境的版本。Kaggle 如果已经有可用的 PyTorch，可仅补充：

```python
%pip install "segmentation-models-pytorch>=0.5,<0.6" PyYAML tqdm matplotlib
```

查看 GPU 是否可用：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

## 3. 数据与划分

这是语义分割任务，共 21 类：`0` 是背景，`1~20` 是物体类别，`255` 是忽略区域。类别名称按编号保存在 `dataset.py` 的 `CLASS_NAMES` 中。

VOC 2012 的公开训练/验证数据包含有标签的 `train` 和 `val`。官方竞赛测试标签不公开，Torchvision 的 2012 版本不能设置 `image_set="test"`。

本项目使用下面的划分，训练和选择模型时不会读取最终测试图片：

| 本项目用途 | 数据来源 | 默认图片数 |
| --- | --- | ---: |
| 训练 | 官方 train 的 90% | 1318 |
| 内部验证：选择最佳模型 | 官方 train 的 10% | 146 |
| 最终本地测试 | 官方 val | 1449 |

划分由 `seed` 和 `val_ratio` 决定，具体图片名保存在 `outputs/split.json`。这里的测试是本项目的保留集评估，不是官方 VOC test 榜单成绩。

首次运行默认自动下载。Torchvision 下载的是整个 VOC 2012 train/val 压缩包，约 2 GB；下载和解压需要时间及额外磁盘空间。下载成功后可将 `download` 改为 `false`。

已有数据时，`data_root` 必须指向包含 `VOCdevkit` 的上一级目录。例如：

| 内容 | 示例路径 |
| --- | --- |
| config 中的 data_root | `./data` |
| RGB 图片 | `./data/VOCdevkit/VOC2012/JPEGImages/` |
| 语义分割标签 | `./data/VOCdevkit/VOC2012/SegmentationClass/` |
| train/val 名单 | `./data/VOCdevkit/VOC2012/ImageSets/Segmentation/` |

如果自动下载失败，可自行准备完整 VOC 2012 train/val 数据并保持以上结构，再设置 `download: false`。Kaggle 的数据目录同理；注意 `data_root` 不应直接指向 `VOC2012`。

## 4. 训练和测试

先训练，再加载最佳模型测试：

```bash
python train.py --config config.yaml
python test.py --config config.yaml
```

在 Kaggle / Jupyter 中，先进入存放这些文件的文件夹，然后运行：

```python
%cd /kaggle/working/VOC2012Segmentation
!python -u train.py --config config.yaml
!python -u test.py --config config.yaml
```

路径按你实际上传的位置修改；上面的 `%cd` 是示例。

也可以指定已有权重：

```bash
python test.py --config config.yaml --checkpoint outputs/best_model.pth
```

测试时，模型结构和图片大小读取权重中保存的训练配置，数据路径、设备、batch size 和输出位置读取当前 YAML。权重路径以当前工作目录为起点；YAML 中的数据及输出相对路径以 YAML 所在目录为起点。

第一次想快速检查流程，可以先将 `epochs` 改成 `1`；显存不够时将 `batch_size` 从 `8` 降到 `4` 或 `2`。`device: auto` 会优先使用 CUDA，否则使用 CPU。

每次训练会重新开始，并覆盖同一 `output_dir` 下的训练记录及最佳模型。需要保留不同实验时，请设置不同输出目录。固定随机种子便于复现数据划分，不保证不同设备和软件版本上的数值完全一致。

## 5. 关键代码与形状

假设 `batch_size=8`、`image_size=[256, 256]`：

| 数据 | 形状 | 内容 |
| --- | --- | --- |
| images | `(8, 3, 256, 256)` | float32，RGB 数值范围 `[0, 1]` |
| masks | `(8, 256, 256)` | int64，类别编号 `0~20` 或 `255` |
| logits | `(8, 21, 256, 256)` | 模型输出，每个像素有 21 个分数 |
| predictions | `(8, 256, 256)` | 每个像素预测的类别编号 |

标签图片是带调色板的类别索引图。必须保留原始编号，不能转成 RGB 或除以 255；缩放标签要用最近邻插值。只有输入图片缩放到 `[0, 1]`。

多类别分割使用交叉熵，预测时沿类别维取最大值：

```python
criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="sum")
logits = model(images)
loss = criterion(logits, masks) / (masks != 255).sum()
predictions = logits.argmax(dim=1)
```

`CrossEntropyLoss` 直接接收 logits，不要在模型最后加 softmax。这里不使用 sigmoid 加 0.5 阈值，因为每个像素需要从 21 类中选择一类。

## 6. SMP 指标

代码实际调用：

```python
tp, fp, fn, tn = smp.metrics.get_stats(
    predictions.cpu(), masks.cpu(),
    mode="multiclass", num_classes=21, ignore_index=255,
)
```

每个 batch 得到的计数先累加到整个数据集，再调用 `smp.metrics.iou_score`、`f1_score`、`precision`、`recall`，统一设置 `reduction="macro", zero_division=0`。不会先计算每个 batch 的分数再简单求平均。

| 输出名称 | 含义 |
| --- | --- |
| loss | 所有有效像素的平均交叉熵，越低越好 |
| miou | 21 类 IoU 的平均值，包含背景 |
| f1_score | 21 类 F1/Dice 的平均值，包含背景 |
| precision | 21 类精确率的平均值 |
| recall | 21 类召回率的平均值 |
| pixel_accuracy | 预测正确的有效像素 / 所有有效像素 |

分数以 `0~1` 保存，例如 `0.65` 表示 `65%`。各类别的零分母分数定义为 0；没有真实像素也没有预测像素的类别仍参与 21 类平均，可能降低小样本评估的分数。`255` 不参与损失和指标。

`pixel_accuracy` 直接由 SMP 的计数计算，没有使用 SMP 的多类别 one-vs-rest `accuracy`，因为后者的含义不同。

## 7. 输出文件

| 路径（相对于 output_dir） | 内容 |
| --- | --- |
| `best_model.pth` | 内部验证 mIoU 最高时的模型、配置和 epoch |
| `history.csv` | 每轮 train/val 的 loss、mIoU、F1、precision、recall、像素准确率 |
| `config_used.yaml` | 本次训练实际使用的配置 |
| `split.json` | 训练和验证图片名单 |
| `test_metrics.json` | 最终测试指标与评估设置 |
| `per_class_metrics.csv` | 每个类别单独的 IoU、F1 |
| `predictions/*_comparison.png` | 原图、真实标签、预测标签对比 |
| `predictions/*_mask.png` | 原始预测类别图，每个像素是 `0~20` |

测试指标和预测图均基于配置中的缩放尺寸，不是原始分辨率的官方评测。原始类别图直接查看可能很暗，这是因为灰度数值仅为 `0~20`；彩色效果请看 `comparison.png`。真实标签中的白色表示忽略区域。

小型网络从零学习 VOC 的 21 类有一定难度，前几轮预测偏向背景、指标偏低都可能发生。本例用于学习完整流程，不承诺特定准确率。

## 8. 验证范围

已使用 PyTorch 2.6.0 CPU、Torchvision 0.21.0、SMP 0.5.0，在少量合成的 VOC 格式图片上跑通两轮训练、验证、保存最佳模型、重新加载测试，以及 CSV、JSON、预测图输出。还核对了忽略像素、指标计数、不同 batch size 的评估一致性和非整齐尺寸输入。

这里没有执行完整 VOC 2012 训练，也没有提供训练好的 VOC 权重或宣称真实数据集上的指标；下载代码后由你运行训练。

## 9. 官方参考

- [Torchvision VOCSegmentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.VOCSegmentation.html)
- [VOC 2012 数据与标签说明](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html)
- [VOC 2012 数据量统计](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/dbstats.html)
- [SMP 指标 API](https://smp.readthedocs.io/en/latest/metrics.html)
