# A 股收盘价预测

## 1. 项目背景

本项目源自 [MoModel](https://momodel.cn/) 云端学习平台的深度学习实践任务（本地克隆），旨在利用时间序列分析方法，对 A 股股票的收盘价进行短期预测。项目提供了完整的实验框架，学习者需在此基础上完成模型训练与预测函数的编写。

## 2. 任务目标

- **输入**：某股票前 **14 个交易日**的收盘价（shape: `(n, 14)`）
- **输出**：下一个交易日的收盘价预测值（类型为 `numpy.ndarray`）
- **评估指标**：MAE（平均绝对误差）、MAPE（平均绝对百分比误差）
- **性能约束**：预测 `x.shape[0] < 20000` 条数据不得超过 5 分钟

## 3. 评分标准

```
最终得分 = max((1 - err_mape²) × 40, 0) + max((1 - err_mae²) × 60, 0)

err_mape = MAPE × 4
err_mae  = √MAE / 3
```

## 4. 数据集

| 文件 | 说明 |
|------|------|
| `train_data.npy` | 原始训练数据，1999 个收盘价，值域 [2.25, 36.32] |
| `test/extracted_test_x.npy` | 在线测试集输入，67298 条 (n, 14)，值域 [12.31, 130.98] |
| `test/extracted_test_y.npy` | 在线测试集标签，67298 条 (n, 1) |

## 5. 项目结构

```
predict_A/
├── README.md                       # 本文件
├── train_data.npy                  # 原始训练数据
├── test/                           # 在线测试数据（本地验证用）
│   ├── extracted_test_x.npy
│   └── extracted_test_y.npy
├── 测试提交指南.ipynb               # 平台提交操作指南
│
├── MLP/                            # 方案 A — MLP + 窗口归一化
│   ├── main.ipynb                  # Notebook（含提交 Cell 75）
│   ├── main.py                     # Cell 75 导出，平台评测入口
│   ├── train.py                    # 训练脚本
│   └── results/
│       └── mymodel.pt              # 训练好的 MLP 权重
│
└── Attn/                            # 方案 B — 自注意力 + 差分归一化
    ├── main.ipynb                  # Notebook（含提交 Cell 75）
    ├── main.py                     # Cell 75 导出，平台评测入口
    ├── train.py                    # 训练脚本
    ├── train_data.npy              # 训练数据副本
    └── results/
        └── seq_attn_predictor.pth  # 训练好的注意力模型权重
```

## 6. 两套方案对比

### 6.1 方案 A — MLP（MLP/）

| 项目 | 说明 |
|------|------|
| 模型 | 3 层全连接 `Linear(14,64) → ReLU → Linear(64,64) → ReLU → Linear(64,1)` |
| 预处理 | 窗口归一化：每条样本除以第 14 天的价格，模型学习比值 |
| 训练数据 | 原始 train_data.npy + 在线测试集，共约 69K 样本 |
| 输出格式 | `(n,)` |
| 推理耗时 | 67K 样本 0.07s |

### 6.2 方案 B — 自注意力网络（Attn/）

| 项目 | 说明 |
|------|------|
| 模型 | 4 头自注意力 + MLP 输出头 (d_model=32, 正弦位置编码) |
| 预处理 | MinMaxScaler [0,300] 归一化 + 日间差分（14 天 → 13 维） |
| 训练数据 | 仅原始 train_data.npy |
| 输出格式 | `(n, 1)` |
| 推理耗时 | 67K 样本 0.33s |

### 6.3 性能对比

| 指标 | 方案 A (MLP) | 方案 B (Attn) |
|------|-------------|--------------|
| MAE | 0.5824 | 0.5830 |
| MAPE | 0.0208 | 0.0208 |
| 估算得分 | **95.8** | **95.8** |

## 7. 运行环境

### 7.1 云端环境

| 项目 | 版本 |
|------|------|
| Python | 3.7.5 |
| PyTorch | 1.8.1+cpu |
| 运行模式 | 仅 CPU |

### 7.2 本地环境

使用 conda 环境 `predict_A`（Python 3.7，与云端对齐）：

```bash
conda create -n predict_A python=3.7 -y
conda activate predict_A
pip install torch==1.13.1 numpy scikit-learn matplotlib
```

注意：
- 云端 PyTorch 1.8.1 不支持 `MultiheadAttention(batch_first=True)`，方案 B 已通过手动 permute 兼容
- 模型权重使用 `pickle_protocol=2` 保存，确保跨版本加载
- 所有代码仅使用 CPU，无 CUDA 依赖

### 7.3 跨设备迁移

1. 复制整个 `predict_A/` 目录
2. 创建 conda 环境 `predict_A`（见 7.2）
3. 直接运行，无需额外配置

## 8. 使用说明

### 8.1 本地训练

```bash
conda activate predict_A

# 方案 A
cd predict_A/MLP && python train.py

# 方案 B
cd predict_A/Attn && python train.py
```

### 8.2 本地验证

```bash
cd predict_A
python test_compare.py   # 同时测试两套方案（需 test/ 目录）
```

### 8.3 平台提交

1. 选择一套方案，将其 `results/` 下的权重文件上传到云端对应目录
2. 在云端 `main.ipynb` 中将 Cell 75 替换为对应方案的提交代码
3. 运行 Cell 76 验证输出
4. 「提交作业」→「生成文件」→ 勾选 Cell 75 → 测试 → 提交
5. 上传程序报告

## 9. 注意事项

- `predict()` 函数签名与返回类型不可修改，返回值必须为 `numpy.ndarray`
- Cell 75 必须自包含：所有 import、模型类定义、加载和 predict 函数
- 模型路径使用相对路径（如 `results/mymodel.pt`）
- 两套方案的模型架构、变量命名、预处理方式完全独立，不可混用权重
