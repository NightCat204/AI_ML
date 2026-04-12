# -*- coding: utf-8 -*-
"""股价预测模型训练脚本 —— 基于自注意力机制的日间差分预测"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 可复现性设置
# ============================================================
def fix_random_state(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_random_state(42)


# ============================================================
# 数据读取与预处理
# ============================================================
print("加载训练数据 ...")
raw_prices = np.load('train_data.npy')

# 统一使用固定区间 [0, 300] 做 MinMax 归一化
normalizer = MinMaxScaler().fit(np.array([0, 300]).reshape(-1, 1))
prices_normed = normalizer.transform(raw_prices.reshape(-1, 1)).flatten()


def build_delta_samples(series, window=14):
    """
    根据归一化后的价格序列，通过滑动窗口构造差分样本。
    输入：长度为 window 的子序列，计算相邻日差分得到 window-1 维特征向量。
    标签：窗口结束后下一日与窗口最后一日的差值（日间增量）。
    同时记录窗口最后一日的归一化价格，用于评估时还原真实价格。
    """
    delta_x, delta_y, anchor = [], [], []
    for t in range(window, len(series)):
        seg = series[t - window:t]
        delta_x.append(np.diff(seg))
        delta_y.append(series[t] - series[t - 1])
        anchor.append(series[t - 1])
    return np.array(delta_x), np.array(delta_y), np.array(anchor)


delta_x, delta_y, anchor_vals = build_delta_samples(prices_normed)


def split_dataset(x, y, anc, ratios=(0.7, 0.1, 0.2)):
    """按比例切分训练 / 验证 / 测试集"""
    n = x.shape[0]
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])

    slices = {}
    slices['train'] = (x[:n_train], y[:n_train], anc[:n_train])
    slices['val']   = (x[n_train:n_train + n_val],
                       y[n_train:n_train + n_val],
                       anc[n_train:n_train + n_val])
    slices['test']  = (x[n_train + n_val:],
                       y[n_train + n_val:],
                       anc[n_train + n_val:])
    return slices


splits = split_dataset(delta_x, delta_y, anchor_vals)

# 构建 DataLoader
def make_loader(xy, batch, shuffle):
    x_t = torch.tensor(xy[0], dtype=torch.float32)
    y_t = torch.tensor(xy[1], dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch, shuffle=shuffle)

train_loader = make_loader(splits['train'], batch=32, shuffle=True)
val_loader   = make_loader(splits['val'],   batch=32, shuffle=False)


# ============================================================
# 模型定义
# ============================================================
class SeqAttnPredictor(nn.Module):
    """基于自注意力机制的日间增量预测网络"""

    def __init__(self, seq_len=13, d_model=32, n_heads=4):
        super(SeqAttnPredictor, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.proj = nn.Linear(seq_len, seq_len * d_model)
        self.register_buffer('pe', self._sincos_pe(seq_len, d_model))
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _sincos_pe(length, dim):
        pos = np.arange(length)[:, None]
        idx = np.arange(dim)[None, :]
        angle = pos / np.power(10000, 2 * (idx // 2) / dim)
        pe = np.zeros_like(angle)
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        return torch.FloatTensor(pe).unsqueeze(0)

    def forward(self, x):
        bsz = x.size(0)
        h = self.proj(x).view(bsz, self.seq_len, self.d_model)
        h = h + self.pe[:, :self.seq_len, :]
        h = h.permute(1, 0, 2)
        h, _ = self.self_attn(h, h, h)
        h = h.permute(1, 0, 2).mean(dim=1)
        return self.head(h)


hw_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SeqAttnPredictor().to(hw_device)


# ============================================================
# 评估函数
# ============================================================
def calc_mae(pred, real):
    return torch.mean(torch.abs(pred - real))

def calc_mape(pred, real, eps=1e-8):
    return torch.mean(torch.abs(pred - real) / (real + eps))

def calc_rmse(pred, real):
    return torch.sqrt(torch.mean((pred - real) ** 2))

def run_eval(loader, model, scaler, anchor_arr):
    """在原始价格尺度上计算 MAE / MAPE / RMSE"""
    accum_mae, accum_mape, accum_rmse, total = 0., 0., 0., 0
    model.eval()
    with torch.no_grad():
        cursor = 0
        for bx, by in loader:
            bx, by = bx.to(hw_device), by.to(hw_device)
            pred_delta = model(bx)

            bsz = bx.size(0)
            anc = anchor_arr[cursor:cursor + bsz]
            cursor += bsz

            # 还原归一化尺度的预测值和真实值
            pred_normed = anc + pred_delta.cpu().numpy().flatten()
            real_normed = anc + by.cpu().numpy().flatten()

            pred_price = scaler.inverse_transform(pred_normed.reshape(-1, 1)).flatten()
            real_price = scaler.inverse_transform(real_normed.reshape(-1, 1)).flatten()

            pt = torch.tensor(pred_price, dtype=torch.float32).to(hw_device)
            rt = torch.tensor(real_price, dtype=torch.float32).to(hw_device)

            accum_mae  += calc_mae(pt, rt).item() * bsz
            accum_mape += calc_mape(pt, rt).item() * bsz
            accum_rmse += calc_rmse(pt, rt).item() * bsz
            total += bsz

    model.train()
    return accum_mae / total, accum_mape / total, accum_rmse / total


# ============================================================
# 训练循环
# ============================================================
loss_fn   = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

N_EPOCHS = 100
best_mae  = float('inf')
save_path = "results/seq_attn_predictor.pth"
os.makedirs("results", exist_ok=True)

header = "{:<6}{:<12}{:<12}{:<12}{:<12}{:<12}{:<12}".format(
    "Epoch", "Tr Loss", "Va Loss", "Tr MAE", "Va MAE", "Va RMSE", "Va MAPE")
print(header)
print("-" * len(header))

for epoch in range(1, N_EPOCHS + 1):
    net.train()
    running_loss = 0.
    for bx, by in train_loader:
        bx, by = bx.to(hw_device), by.to(hw_device)
        optimizer.zero_grad()
        out = net(bx)
        batch_loss = loss_fn(out, by)
        batch_loss.backward()
        optimizer.step()
        running_loss += batch_loss.item()

    # 验证损失
    v_loss = 0.
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(hw_device), by.to(hw_device)
            v_loss += loss_fn(net(bx), by).item()

    tr_mae, tr_mape, tr_rmse = run_eval(
        train_loader, net, normalizer, splits['train'][2])
    va_mae, va_mape, va_rmse = run_eval(
        val_loader, net, normalizer, splits['val'][2])

    if va_mae < best_mae:
        best_mae = va_mae
        torch.save(net.state_dict(), save_path)
        print("Epoch {}: 最优模型已保存, Val MAE = {:.6f}".format(epoch, va_mae))

    if epoch % 10 == 0:
        print("{:<6}{:<12.6f}{:<12.6f}{:<12.6f}{:<12.6f}{:<12.6f}{:<12.6f}".format(
            epoch,
            running_loss / len(train_loader),
            v_loss / len(val_loader),
            tr_mae, va_mae, va_rmse, va_mape))

print("训练结束, 最优 Val MAE: {:.6f}".format(best_mae))
print("权重保存于: {}".format(save_path))
