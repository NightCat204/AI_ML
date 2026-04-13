# A 股收盘价预测

## 1. 项目概述

本项目是基于 MoModel A 股收盘价预测任务整理出的本地工程，当前仓库中同时保留了两套实现方案：

- `MLP/`：窗口归一化 + 多层感知机
- `Attn/`：固定尺度归一化 + 差分序列 + 轻量自注意力

除模型代码外，仓库还包含本地测试数据、平台提交用 notebook、报告源码与 PDF，以及一个单独归档到 `Docs/` 的外部参考材料文件。

## 2. 任务定义

- 输入：某股票前 14 个交易日的收盘价，形状为 `(n, 14)`
- 输出：第 15 个交易日的预测收盘价，返回类型必须为 `numpy.ndarray`
- 评估指标：MAE、MAPE
- 评测约束：当 `x.shape[0] < 20000` 时，单次预测耗时不得超过 5 分钟

评分公式如下：

```text
Score = max((1 - err_mape^2) * 40, 0) + max((1 - err_mae^2) * 60, 0)

err_mape = 4 * MAPE
err_mae  = sqrt(MAE) / 3
```

## 3. 当前数据文件

下表反映的是当前仓库中实际存在的数据文件，而不是平台抽象描述：

| 文件                          | 当前状态                                                          |
| ----------------------------- | ----------------------------------------------------------------- |
| `train_data.npy`            | 形状 `(1999,)`，`float64`，价格范围 `[2.25, 36.32]`         |
| `test/extracted_test_x.npy` | 形状 `(67298, 14)`，`float32`，价格范围约 `[12.31, 130.98]` |
| `test/extracted_test_y.npy` | 形状 `(67298, 1)`，`float32`，价格范围约 `[12.31, 130.98]`  |

## 4. 当前目录结构

以下结构基于当前仓库内容整理，省略了 `__pycache__`、`.DS_Store` 和部分 LaTeX 编译辅助文件：

```text
predict_A/
├── README.md
├── train_data.npy
├── test/
│   ├── extracted_test_x.npy
│   └── extracted_test_y.npy
├── 测试提交指南.ipynb
├── Docs/
│   └── xjg报告.pdf
├── MLP/
│   ├── main.ipynb
│   ├── main.py
│   ├── train.py
│   └── results/
│       ├── mymodel.pt
│       ├── README.md
│       └── tb_results/
│           └── README.md
├── Attn/
│   ├── main.ipynb
│   ├── main.py
│   ├── train.py
│   └── results/
│       ├── seq_attn_predictor.pth
│       ├── README.md
│       └── tb_results/
│           └── README.md
└── Report/
    ├── thesis.tex
    ├── thesis.pdf
    ├── bibfile.bib
    ├── gbt7714-2005.bst
    ├── body/
    └── figure/
```

其中：

- `Docs/xjg报告.pdf` 是单独归档的外部参考材料，不属于运行依赖。
- `Report/` 是当前课程报告工程，包含 LaTeX 源码、图片和已生成的 PDF。
- 两个 `results/` 目录下都已经有现成权重文件，不需要重新训练才能查看提交入口代码。

## 5. 两套方案的当前实现

### 5.1 方案 A：`MLP/`

当前文件职责如下：

- `MLP/main.py`
  - 平台提交入口
  - 定义 `MLPNet`
  - 通过相对路径 `results/mymodel.pt` 加载权重
  - `predict(test_x)` 返回形状 `(n,)`
- `MLP/train.py`
  - 训练脚本源码
  - `load_all_data()` 会优先读取当前工作目录下的 `train_data.npy`
  - 如果当前工作目录下存在 `test/extracted_test_x.npy` 和 `test/extracted_test_y.npy`，会一并并入训练
  - 训练策略为窗口归一化 + MSE + Adam + `ReduceLROnPlateau` + 早停

模型结构为：

```text
Linear(14, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 1)
```

### 5.2 方案 B：`Attn/`

当前文件职责如下：

- `Attn/main.py`
  - 平台提交入口
  - 定义 `SeqAttnPredictor`
  - 通过相对路径 `./results/seq_attn_predictor.pth` 加载权重
  - `predict(test_x)` 返回形状 `(n, 1)`
- `Attn/train.py`
  - 训练脚本源码
  - 从当前工作目录读取 `train_data.npy`
  - 用固定区间 `[0, 300]` 做 `MinMaxScaler`
  - 将长度 14 的价格窗口转换为长度 13 的差分输入
  - 使用 4 头自注意力与位置编码预测下一步增量

该方案当前只使用 `train_data.npy` 构造训练、验证和测试划分，不会读取 `test/` 下的数据。

## 6. 当前保存的权重与结果

当前仓库中已经存在以下权重文件：

- `MLP/results/mymodel.pt`
- `Attn/results/seq_attn_predictor.pth`

当前文档中记录的结果为：

| 指标     | MLP    | Attn   |
| -------- | ------ | ------ |
| MAE      | 0.5824 | 0.5830 |
| MAPE     | 0.0208 | 0.0208 |
| 估算得分 | 95.8   | 95.8   |

## 7. 运行环境与依赖

从当前脚本依赖看，核心环境要求如下：

- Python 3.7
- NumPy
- PyTorch
- scikit-learn

若按本地 conda 环境复现，可使用：

```bash
conda create -n predict_A python=3.7 -y
conda activate predict_A
pip install numpy torch scikit-learn
```

说明：

- 方案 B 的实现显式兼容不支持 `batch_first=True` 的旧版 `MultiheadAttention`
- 代码全部以 CPU 为目标编写
- MLP 权重保存使用了 `pickle_protocol=2`

## 8. 当前路径依赖与可运行性说明

这一部分是当前 README 最需要更新的地方。

### 8.1 `main.py` 的路径假设

两套提交入口都通过相对路径加载权重：

- `MLP/main.py` 读取 `results/mymodel.pt`
- `Attn/main.py` 读取 `./results/seq_attn_predictor.pth`

因此，若直接在本地调用 `main.py`，当前工作目录应与对应方案目录匹配，或者需要保证权重文件以相同相对路径可见。

### 8.2 `train.py` 的路径假设

两份训练脚本都通过相对路径读取数据：

- `MLP/train.py` 读取 `train_data.npy`，并在存在时读取 `test/extracted_test_x.npy` 与 `test/extracted_test_y.npy`
- `Attn/train.py` 读取 `train_data.npy`

但在当前仓库结构下，共享数据文件位于项目根目录 `predict_A/`，而权重目录位于 `MLP/results/` 与 `Attn/results/`。因此：

- README 旧版本中写的 `cd predict_A/MLP && python train.py` 和 `cd predict_A/Attn && python train.py` 并不准确
- 按当前仓库状态，若要直接复现训练，需要先处理相对路径问题，例如调整脚本中的路径、调整工作目录，或临时布置数据文件位置

换句话说，当前仓库更准确的状态是：

- `main.py` 与现成权重可直接作为提交入口源码查看
- `train.py` 保留了完整训练逻辑，但不是一个已经对当前目录结构完全对齐的一键训练脚本

## 9. 其他说明

- `测试提交指南.ipynb`、`MLP/main.ipynb` 与 `Attn/main.ipynb` 仍保留，主要用于平台提交流程与 notebook 版本代码
- 两套方案的模型结构、预处理方式和权重文件彼此独立，不能混用
