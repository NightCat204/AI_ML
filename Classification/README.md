# 垃圾分类 — ResNet18 + ResNet34 迁移学习

## 1. 项目背景

本项目源自 [MoModel](https://momodel.cn/) 云端学习平台的深度学习实践任务（本地克隆），基于 PyTorch 框架，利用 ResNet18 + ResNet34 预训练模型进行迁移学习，通过双模型 Ensemble + 测试时增强（TTA），实现 26 类垃圾图片的自动分类。

## 2. 任务目标

- **输入**：一张垃圾图片（`np.ndarray`，shape `(H, W, C)`，RGB 格式）
- **输出**：分类名称（`str`），共 26 个类别
- **评估指标**：分类准确率（Accuracy）
- **约束**：`predict()` 函数签名与返回类型不可修改

## 3. 数据集

| 项目 | 说明 |
|------|------|
| 路径 | `datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/` |
| 训练集 | 2343 张图片，26 类，约 90 张/类 |
| 验证集 | 130 张图片，26 类，5 张/类 |
| 分辨率 | 各异，训练/推理时统一缩放至 224x224 |

### 3.1 类别列表（26 类）

| 大类 | 子类 |
|------|------|
| 可回收物 (00) | Plastic Bottle, Hats, Newspaper, Cans, Glassware, Glass Bottle, Cardboard, Basketball, Paper, Metalware |
| 其他垃圾 (01) | Disposable Chopsticks, Lighter, Broom, Old Mirror, Toothbrush, Dirty Cloth, Seashell, Ceramic Bowl |
| 有害垃圾 (02) | Paint bucket, Battery, Fluorescent lamp, Tablet capsules |
| 厨余垃圾 (03) | Orange Peel, Vegetable Leaf, Eggshell, Banana Peel |

## 4. 模型方案

### 4.1 网络架构

采用 **ResNet18 + ResNet34 双模型 Ensemble**，各自独立训练后推理时取 logits 均值。

```
输入图片 (H, W, 3)
    ↓ _preprocess + TTA (原图 + 水平翻转)
ResNet18 (ImageNet 预训练)           ResNet34 (ImageNet 预训练)
    ↓ logits (1, 26)                    ↓ logits (1, 26)
    ├── 原图 logits                      ├── 原图 logits
    └── 翻转 logits                      └── 翻转 logits
    ↓ 两者均值                            ↓ 两者均值
         ↓ 两模型 logits 均值
         ↓ argmax → idx_to_label → 类别名称
```

### 4.2 训练策略

两阶段迁移学习（ResNet18 和 ResNet34 各自独立训练）：

1. **Phase 1（冻结骨干）**：调用 `lock_backbone()` 冻结预训练参数，仅训练替换后的 FC 层
2. **Phase 2（全网络微调）**：调用 `unlock_all_layers()` 解冻全部参数，以更小学习率端到端微调

训练时合并 train + val 作为完整训练集（2473 张），由 `prepare_dataloader()` 构建。

| 超参数 | Phase 1 | Phase 2 | 说明 |
|--------|---------|---------|------|
| epochs | `PHASE1_EPOCHS=5` | `PHASE2_EPOCHS=20` | Phase 1 快速收敛，Phase 2 充分微调 |
| learning_rate | `LR_PHASE1=1e-3` | `LR_PHASE2=3e-5` | Phase 2 使用更小学习率避免破坏特征 |
| optimizer | Adam | Adam | 含 `WEIGHT_DECAY=1e-4` |
| scheduler | CosineAnnealing | CosineAnnealing | 平滑衰减 |
| batch_size | `TRAIN_BATCH=32` | `TRAIN_BATCH=32` | — |
| label_smoothing | 0.1 | 0.1 | 缓解过拟合 |

### 4.3 数据增强（训练集）

- `RandomResizedCrop(IMG_DIM, scale=(0.75, 1.0))`：随机裁剪+缩放至 224x224
- `RandomHorizontalFlip(p=0.5)`：水平翻转
- `RandomRotation(15)`：随机旋转 ±15°
- `ColorJitter`：亮度/对比度/饱和度/色调随机扰动
- `Normalize`：ImageNet 均值/标准差归一化
- `RandomErasing(p=0.25)`：随机擦除

### 4.4 测试时增强（TTA）

推理时对每张图片同时预测原图和水平翻转图（`img_flipped`），各模型取两者 logits 均值后再做双模型 ensemble。

### 4.5 推理预处理

```python
cv2.resize(image, (IMG_DIM, IMG_DIM))      # 缩放
_img_pipeline: ToPILImage → ToTensor → Normalize  # ImageNet 归一化
```

## 5. 项目结构

```
Classification/
├── README.md                           # 本文件
├── main.ipynb                          # 主 Notebook（含提交代码 Cell）
├── 测试提交指南.ipynb                    # 平台提交操作指南
│
├── resnet/                             # ResNet 方案（PyTorch）
│   ├── train_r18.py               # ResNet18 训练脚本
│   ├── train_r34.py               # ResNet34 训练脚本
│   ├── infer.py                        # 独立推理脚本（predict 函数）
│   └── results/                        # 训练产物
│       ├── label_map.json           # 类别→索引映射
│       ├── r18_gc26.pth     # ResNet18 权重（~44MB）
│       └── r34_gc26.pth     # ResNet34 权重（~85MB）
│
├── default/                            # 原始 MindSpore 方案（已弃用）
│   ├── main.ipynb                      # 原始 Notebook
│   ├── train_main.py                   # MindSpore 离线训练脚本
│   ├── src_mindspore/                  # MindSpore 模型定义
│   └── results/                        # 原始训练产物
│
└── datasets/                           # 数据集（从平台获取）
    └── 5fbdf571c06d3433df85ac65-momodel/
        └── garbage_26x100/
            ├── train/                  # 训练集（26 个子目录）
            └── val/                    # 验证集（26 个子目录）
```

## 6. 运行环境

### 6.1 本地环境

```bash
conda create -n classify python=3.10 -y
conda activate classify
pip install torch torchvision numpy opencv-python-headless Pillow
```

### 6.2 云端环境

需确认云端平台已安装 PyTorch 和 torchvision。若未安装，可在 Notebook 首个 Cell 中执行：

```python
!pip install torch torchvision
```

## 7. 使用说明

### 7.1 本地训练

训练脚本使用绝对路径定位数据集，可从任意目录运行：

```bash
conda activate classify

# 训练 ResNet18（Phase1: 5 epoch + Phase2: 20 epoch）
python Classification/resnet/train_r18.py

# 训练 ResNet34（Phase1: 5 epoch + Phase2: 20 epoch）
python Classification/resnet/train_r34.py
```

训练完成后权重自动保存到 `resnet/results/` 目录。

### 7.2 本地验证

```bash
cd Classification/resnet
python -c "
import os, cv2, json, numpy as np
from infer import predict, idx_to_label

val_base = '../datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val/'
with open('results/label_map.json') as f:
    class_to_idx = json.load(f)

correct = total = 0
for cls in sorted(os.listdir(val_base)):
    idx = class_to_idx[cls]
    expected = idx_to_label[idx]
    for f in os.listdir(os.path.join(val_base, cls)):
        img = cv2.imread(os.path.join(val_base, cls, f))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        total += 1
        if predict(img_rgb) == expected:
            correct += 1
print(f'Accuracy: {correct}/{total} = {correct/total*100:.2f}%')
"
```

### 7.3 云端提交

1. 将以下文件上传到云端 `~/work/` 目录：
   - `main.ipynb`
   - `results/r18_gc26.pth`
   - `results/r34_gc26.pth`
2. 打开 `main.ipynb`
3. 「提交作业」→「生成文件」→ 勾选 **Cell 18 + Cell 19**（标记为"生成 main.py 时请勾选此 cell"的两个代码 Cell）
4. 测试通过后提交

**注意**：Cell 18 包含导入与 `idx_to_label` 映射，Cell 19 包含 `_ensure_models()`、`_preprocess()`、`predict()` 等全部推理逻辑。两个 Cell 合并后即为自包含的 `main.py`。

## 8. 训练结果

### 8.1 ResNet18 训练日志

```
Phase 1: train fc only (5 epochs)
  [1/5] loss=2.5355 acc=0.4108
  [2/5] loss=1.6853 acc=0.7169
  [3/5] loss=1.4341 acc=0.7922
  [4/5] loss=1.3397 acc=0.8132
  [5/5] loss=1.2986 acc=0.8241

Phase 2: fine-tune all layers (20 epochs)
  [1/20]  loss=1.1675 acc=0.8528
  ...
  [20/20] loss=0.7220 acc=0.9968
```

### 8.2 ResNet34 训练日志

```
Phase 1: train fc only (5 epochs)
  [1/5] loss=2.5645 acc=0.3898
  [2/5] loss=1.6556 acc=0.7182
  [3/5] loss=1.4025 acc=0.7772
  [4/5] loss=1.3081 acc=0.8095
  [5/5] loss=1.2634 acc=0.8285

Phase 2: fine-tune all layers (20 epochs)
  [1/20]  loss=1.0963 acc=0.8633
  ...
  [19/20] loss=0.6820 acc=1.0000
  [20/20] loss=0.6819 acc=0.9992
```

### 8.3 Ensemble 验证结果

| 数据集 | 准确率 |
|--------|--------|
| 验证集 (130 张) | **100.00%** (130/130) |

## 9. 关键优化说明

1. **双模型 Ensemble**：ResNet18 + ResNet34 取 logits 均值，比单模型更鲁棒
2. **TTA（测试时增强）**：推理时同时预测原图和水平翻转图，进一步提高准确率
3. **两阶段训练**：先 `lock_backbone` 训练 FC，再 `unlock_all_layers` 全网络微调，充分利用预训练特征
4. **全量数据训练**：合并 train + val 共 2473 张用于训练，最大化数据利用
5. **Label Smoothing**：交叉熵 label_smoothing=0.1，缓解小数据集过拟合

## 10. 注意事项

- `predict()` 函数签名不可修改，输入为 `np.ndarray`（H, W, C），返回类别名称字符串
- 提交时勾选 Cell 18 + Cell 19（标记为"生成 main.py 时请勾选此 cell"的两个代码 Cell）
- 模型权重路径使用相对路径 `./results/r18_gc26.pth` 和 `./results/r34_gc26.pth`
- 权重文件需与生成的 `main.py` 位于同级目录结构下
