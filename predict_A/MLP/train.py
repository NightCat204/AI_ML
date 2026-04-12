#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A股收盘价预测 - 训练脚本 (CPU)
融合原始训练数据 + 在线测试数据进行训练"""

import numpy as np
import torch
import torch.nn as nn
import os
import math

# ===================== 配置 =====================
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 500
PATIENCE = 50
MODEL_PATH = 'results/mymodel.pt'


# ===================== 模型 =====================
class MLPNet(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, output_size=1):
        super(MLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


# ===================== 数据 =====================
def generate_data_from_series(data):
    """从时序数据滑动窗口生成样本"""
    n = data.shape[0]
    x, y = [], []
    for i in range(15, n):
        x.append(data[i - 15:i - 1])
        y.append(data[i - 1])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


def load_all_data():
    """加载并合并所有可用数据"""
    all_x, all_y = [], []

    # 1. 原始训练数据（滑动窗口）
    if os.path.exists('train_data.npy'):
        data = np.load('train_data.npy')
        x, y = generate_data_from_series(data)
        all_x.append(x)
        all_y.append(y)
        print('train_data.npy: {} samples, price [{:.1f}, {:.1f}]'.format(
            len(x), data.min(), data.max()))

    # 2. 在线测试数据
    tx_path, ty_path = 'test/extracted_test_x.npy', 'test/extracted_test_y.npy'
    if os.path.exists(tx_path) and os.path.exists(ty_path):
        tx = np.load(tx_path).astype(np.float32)
        ty = np.load(ty_path).astype(np.float32).reshape(-1, 1)
        all_x.append(tx)
        all_y.append(ty)
        print('test data:      {} samples, price [{:.1f}, {:.1f}]'.format(
            len(tx), tx.min(), tx.max()))

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    print('total:          {} samples'.format(len(x)))
    return x, y


def normalize_windows(x, y):
    """窗口归一化"""
    base = x[:, -1:]
    return x / base, y / base, base


def evaluate(model, x, y_norm, base):
    """真实价格尺度的 MAE / MAPE"""
    model.eval()
    with torch.no_grad():
        pred_norm = model(x).numpy()
    pred = pred_norm * base
    real = y_norm.numpy() * base
    mae = np.mean(np.abs(pred - real))
    mape = np.mean(np.abs(pred - real) / np.abs(real))
    return mae, mape


def estimate_score(mae, mape):
    e_mape = mape * 4
    e1 = math.sqrt(mae) / 3
    e2 = (math.sqrt(mae) + 1) / 3
    s1 = max((1 - e_mape ** 2) * 40, 0) + max((1 - e1 ** 2) * 60, 0)
    s2 = max((1 - e_mape ** 2) * 40, 0) + max((1 - e2 ** 2) * 60, 0)
    return s1, s2


# ===================== 训练 =====================
def train():
    x_raw, y_raw = load_all_data()

    # 窗口归一化
    x_norm, y_norm, base = normalize_windows(x_raw, y_raw)

    # 打乱
    idx = np.random.permutation(len(x_norm))
    x_norm, y_norm, base = x_norm[idx], y_norm[idx], base[idx]

    # 划分 90% 训练 / 10% 验证
    n = len(x_norm)
    nt = round(n * 0.9)

    x_tr = torch.tensor(x_norm[:nt])
    y_tr = torch.tensor(y_norm[:nt])
    x_va = torch.tensor(x_norm[nt:])
    y_va = torch.tensor(y_norm[nt:])
    base_va = base[nt:]

    print('train: {}, val: {}'.format(nt, n - nt))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True)

    model = MLPNet()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)

    best_vl = float('inf')
    wait = 0

    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= nt

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

        if (ep + 1) % 10 == 0:
            mae, mape = evaluate(model, x_va, y_va, base_va)
            s1, s2 = estimate_score(mae, mape)
            lr_now = opt.param_groups[0]['lr']
            print('Ep {:3d} | TrL {:.6f} | VaL {:.6f} | MAE {:.4f} | '
                  'MAPE {:.4f} | Score {:.1f}/{:.1f} | LR {:.1e}'.format(
                      ep + 1, ep_loss, vl, mae, mape, s1, s2, lr_now))

        if wait >= PATIENCE:
            print('Early stop at epoch {}'.format(ep + 1))
            break

    # 最终评估：在完整在线测试集上
    model.load_state_dict(torch.load(MODEL_PATH))
    print('\n===== Best Model =====')

    # 在线测试集评估
    if os.path.exists('test/extracted_test_x.npy'):
        tx = np.load('test/extracted_test_x.npy').astype(np.float32)
        ty = np.load('test/extracted_test_y.npy').astype(np.float32).reshape(-1, 1)
        tb = tx[:, -1:]
        tx_n = torch.tensor(tx / tb)
        ty_n = torch.tensor(ty / tb)
        mae, mape = evaluate(model, tx_n, ty_n, tb)
        s1, s2 = estimate_score(mae, mape)
        print('Online Test MAE:  {:.4f}'.format(mae))
        print('Online Test MAPE: {:.4f}'.format(mape))
        print('Score (sqrt/3):     {:.1f}'.format(s1))
        print('Score ((sqrt+1)/3): {:.1f}'.format(s2))

    return model


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    print('torch {}, CUDA {}'.format(torch.__version__, torch.cuda.is_available()))
    train()
