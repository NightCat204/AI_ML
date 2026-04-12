# A 股收盘价预测

## 1. 项目背景

本项目源自 [MoModel](https://momodel.cn/) 云端学习平台的深度学习实践任务（本地克隆），旨在利用时间序列分析方法，对 A 股股票的收盘价进行短期预测。项目提供了完整的实验框架（数据加载、预处理、模型搭建、训练评估、提交规范），学习者需在此基础上完成模型训练与预测函数的编写。

## 2. 任务目标

- **输入**：某股票前 **14 个交易日**的收盘价（shape: `(n, 14)`）
- **输出**：下一个交易日的收盘价预测值（shape: `(n,)`，类型为 `numpy.ndarray`）
- **评估指标**：
  - **MAE**（Mean Absolute Error）：平均绝对误差 `mean(|y_hat - y|)`
  - **MAPE**（Mean Absolute Percentage Error）：平均绝对百分比误差 `mean(|y_hat - y| / y)`
- **性能约束**：预测 `x.shape[0] < 20000` 条数据不得超过 5 分钟

## 3. 评分标准

```
最终得分 = max((1 - err_mape²) × 40, 0) + max((1 - err_mae²) × 60, 0)

其中：
  err_mape = MAPE × 4
  err_mae  = √MAE / 3    （或 (√MAE + 1) / 3，待确认）
```

**得分参考**（以 `err_mae = √MAE / 3` 为例）：

| MAE  | MAPE  | 估算得分 |
|------|-------|---------|
| 0.3  | 0.02  | 98.5    |
| 0.5  | 0.03  | 96.1    |
| 1.0  | 0.05  | 91.1    |
| 2.0  | 0.08  | 82.0    |
| 4.0  | 0.10  | 67.1    |

## 4. 数据集说明

| 文件 | 格式 | 说明 |
|------|------|------|
| `train_data.npy` | NumPy 一维数组 | 1999 个收盘价数据点，值域 [2.25, 36.32]，dtype float64 |

数据通过滑动窗口方式生成训练样本：每 14 个连续交易日为输入 x，第 15 日为标签 y。共生成 1984 个样本，按 70% / 10% / 20% 划分为训练集、校验集、测试集。

## 5. 项目结构

```
predict_A/
├── main.ipynb              # 主 Notebook：实验说明 + 代码框架 + 作业区域
├── train.py                # 本地训练脚本（独立运行）
├── 测试提交指南.ipynb        # 平台测试与提交操作指南
├── train_data.npy           # 训练数据集
├── results/                 # 模型权重与训练结果保存目录
│   ├── tb_results/          # TensorBoard 日志目录
│   └── mymodel.pt           # 训练好的模型权重
└── README.md                # 本文件
```

## 6. 实现方案

### 6.1 数据预处理 — 窗口归一化

采用**窗口归一化**替代全局 MinMaxScaler：以每个窗口最后一天的价格为基准，将整个窗口除以该基准值。

```python
base = x[:, -1:]       # 取最后一天价格作为基准
x_norm = x / base      # 输入归一化，最后一个元素恒为 1.0
y_norm = y / base      # 标签归一化为次日/当日的比值
```

**优势**：
- 模型学习的是**相对价格变化**（比值），而非绝对价格
- 天然适配任意价位区间的股票，无需预设归一化范围
- `predict()` 无需依赖训练数据统计量，自包含且可移植

### 6.2 模型架构 — GRU

选用单层 GRU（Gated Recurrent Unit）网络：

```
输入 (batch, 14) → unsqueeze → (batch, 14, 1)
    → GRU(input=1, hidden=64, layers=1)
    → 取最后时间步 (batch, 64)
    → Linear(64, 1) → 输出 (batch, 1)
```

- **参数量**：约 13K，适配 1984 样本的小数据集
- **结构简洁**：单层 GRU + 全连接，训练快速（CPU 数秒完成）
- **时序建模**：GRU 门控机制捕捉 14 天内的价格变化趋势

### 6.3 训练策略

| 项目 | 配置 |
|------|------|
| 损失函数 | MSELoss |
| 优化器 | Adam (lr=1e-3, weight_decay=1e-4) |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=20) |
| 早停 | patience=80 epochs |
| 最大轮次 | 500 |
| 批大小 | 64 |
| 模型选择 | 保存验证集 loss 最低的模型 |

## 7. 运行环境

### 7.1 云端环境（MoModel 平台）

| 项目 | 版本 |
|------|------|
| Python | 3.7.5 |
| Keras | 2.4.3 |
| scikit-learn | （见平台 `pip list`） |
| PyTorch | 待确认：`python -c "import torch; print(torch.__version__)"` |
| 运行模式 | **仅 CPU** |

### 7.2 本地环境

使用 conda 环境 `predict_A`（Python 3.7，与云端一致）：

```bash
# 创建环境
conda create -n predict_A python=3.7 -y
conda activate predict_A

# 安装依赖（CPU 版 PyTorch）
pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/cpu
pip install numpy scikit-learn matplotlib pandas
```

### 7.3 跨设备迁移

1. 复制整个 `predict_A/` 目录
2. 按 7.2 创建同名 conda 环境
3. 确保 `results/mymodel.pt` 已包含训练好的权重
4. 代码全程基于 CPU，无 CUDA 依赖

## 8. 使用说明

### 8.1 本地训练

```bash
cd predict_A
conda activate predict_A
python train.py
```

训练完成后模型自动保存到 `results/mymodel.pt`。

### 8.2 本地测试预测

```python
import numpy as np
# 在 main.ipynb 中执行 Cell 75 加载模型后：
test_x = np.array([[6.69,6.72,6.52,6.66,6.74,6.55,6.35,6.14,6.18,6.17,5.72,5.78,5.69,5.67]])
print(predict(test_x))
```

### 8.3 平台提交流程

1. 将本地训练好的 `results/mymodel.pt` 上传到云端对应目录
2. 在 `main.ipynb` 中运行 Cell 75（模型预测代码答题区域）
3. 运行 Cell 76 验证预测输出
4. 点击「提交作业」->「生成文件」，勾选 Cell 75
5. 生成测试文件 -> 测试 -> 提交
6. 上传「程序报告.docx」或「程序报告.pdf」

## 9. 注意事项

- `predict()` 函数签名与返回类型**不可修改**，返回值必须为 `numpy.ndarray`
- Cell 75（提交单元格）必须**自包含**：包含所有 import、模型类定义、模型加载和 predict 函数
- 模型路径使用相对路径 `results/mymodel.pt`
- 确保本地 PyTorch 版本与云端兼容，避免模型加载失败
- 预测指标不应低于 LinearNet 基线模型

## 10. 待确认信息

1. **云端 PyTorch 版本**（执行 `python -c "import torch; print(torch.__version__)"`）
2. 提交截止时间
3. 程序报告模板或格式要求

## 11. 参考资料

- [PyTorch 教程](https://pytorch.org/tutorials/)
- [《动手学深度学习》PyTorch 版](https://tangshusen.me/Dive-into-DL-PyTorch/)
- [scikit-learn 文档](https://scikit-learn.org/stable/)
