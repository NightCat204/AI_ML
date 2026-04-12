#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A股收盘价预测 - 训练脚本 (CPU)"""

import numpy as np
import torch
import torch.nn as nn
import os
import math

# ===================== 配置 =====================
WINDOW_SIZE = 14
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 500
PATIENCE = 80
MODEL_PATH = 'results/mymodel.pt'
DATA_PATH = 'train_data.npy'


# ===================== 模型 =====================
class GRUNet(nn.Module):
    """GRU 股价预测模型：输入 (batch, 14) 归一化序列，输出 (batch, 1) 预测比值"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(2)          # (batch, 14, 1)
        out, _ = self.gru(x)            # (batch, 14, hidden)
        return self.fc(out[:, -1, :])   # (batch, 1)


# ===================== 数据 =====================
def generate_data(data):
    """滑动窗口生成样本，与 main.ipynb 一致"""
    n = data.shape[0]
    x, y = [], []
    for i in range(15, n):
        x.append(data[i - 15:i - 1])
        y.append(data[i - 1])
    return np.array(x), np.array(y)


def evaluate(model, x, y_norm, base):
    """在真实价格尺度上计算 MAE / MAPE"""
    model.eval()
    with torch.no_grad():
        pred_norm = model(x).numpy()
    pred = pred_norm * base
    real = y_norm.numpy() * base
    mae = np.mean(np.abs(pred - real))
    mape = np.mean(np.abs(pred - real) / np.abs(real))
    return mae, mape


def estimate_score(mae, mape):
    """估算平台得分（两种 err_mae 解释）"""
    e_mape = mape * 4
    e_mae_v1 = math.sqrt(mae) / 3                # 解释1
    e_mae_v2 = (math.sqrt(mae) + 1) / 3          # 解释2
    s1 = max((1 - e_mape ** 2) * 40, 0) + max((1 - e_mae_v1 ** 2) * 60, 0)
    s2 = max((1 - e_mape ** 2) * 40, 0) + max((1 - e_mae_v2 ** 2) * 60, 0)
    return s1, s2


# ===================== 训练 =====================
def train():
    data = np.load(DATA_PATH)
    print('Data: {} points, range [{:.2f}, {:.2f}]'.format(
        data.shape[0], data.min(), data.max()))

    # 生成样本 & 窗口归一化（以每个窗口最后一天价格为基准）
    x, y = generate_data(data)
    base = x[:, -1:].astype(np.float32)       # (n, 1)
    x_norm = (x / base).astype(np.float32)    # (n, 14)
    y_norm = (y.reshape(-1, 1) / base).astype(np.float32)  # (n, 1)

    # 划分 70% / 10% / 20%
    n = len(x)
    nt = round(n * 0.7)
    nv = round(n * 0.1)

    x_tr = torch.tensor(x_norm[:nt])
    y_tr = torch.tensor(y_norm[:nt])
    x_va = torch.tensor(x_norm[nt:nt + nv])
    y_va = torch.tensor(y_norm[nt:nt + nv])
    x_te = torch.tensor(x_norm[nt + nv:])
    y_te = torch.tensor(y_norm[nt + nv:])
    base_te = base[nt + nv:]

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True)

    model = GRUNet(1, HIDDEN_SIZE, NUM_LAYERS, 1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)

    best_vl = float('inf')
    wait = 0

    for ep in range(EPOCHS):
        # 训练
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= nt

        # 验证
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(x_va), y_va).item()
        sched.step(vl)

        if vl < best_vl:
            best_vl = vl
            wait = 0
            torch.save(model.state_dict(), MODEL_PATH, pickle_protocol=2)
        else:
            wait += 1

        if (ep + 1) % 50 == 0:
            mae, mape = evaluate(model, x_te, y_te, base_te)
            s1, s2 = estimate_score(mae, mape)
            lr_now = opt.param_groups[0]['lr']
            print('Ep {:4d} | TrL {:.6f} | VaL {:.6f} | MAE {:.4f} | '
                  'MAPE {:.4f} | Score {:.1f}/{:.1f} | LR {:.1e}'.format(
                      ep + 1, ep_loss, vl, mae, mape, s1, s2, lr_now))

        if wait >= PATIENCE:
            print('Early stop at epoch {}'.format(ep + 1))
            break

    # 最终评估
    model.load_state_dict(torch.load(MODEL_PATH))
    mae, mape = evaluate(model, x_te, y_te, base_te)
    s1, s2 = estimate_score(mae, mape)
    print('\n===== Best Model =====')
    print('MAE:  {:.4f}'.format(mae))
    print('MAPE: {:.4f}'.format(mape))
    print('Score (sqrt(MAE)/3):       {:.1f}'.format(s1))
    print('Score ((sqrt(MAE)+1)/3):   {:.1f}'.format(s2))
    return model


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    print('torch {}, CUDA {}'.format(torch.__version__, torch.cuda.is_available()))
    train()
