# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 差分序列长度 = 14 天收盘价产生 13 个日间变化量
SEQ_LEN = 13
OUT_DIM = 1


class SeqAttnPredictor(nn.Module):
    """基于自注意力机制的股价日间变化量预测网络"""

    def __init__(self, seq_len=13, d_model=32, n_heads=4):
        super(SeqAttnPredictor, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # 线性映射：将输入展开为 seq_len 个 d_model 维的 token
        self.proj = nn.Linear(seq_len, seq_len * d_model)

        # 固定的三角函数位置编码
        self.register_buffer('pe', self._sincos_pe(seq_len, d_model))

        # 多头自注意力（兼容 PyTorch 1.8，不使用 batch_first）
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)

        # 输出头
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _sincos_pe(length, dim):
        """构建正余弦位置编码矩阵"""
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

        # (B, S, D) -> (S, B, D) 适配 MultiheadAttention 默认输入格式
        h = h.permute(1, 0, 2)
        h, _ = self.self_attn(h, h, h)
        # (S, B, D) -> (B, S, D) -> 均值池化 -> (B, D)
        h = h.permute(1, 0, 2).mean(dim=1)

        return self.head(h)


# 实例化并加载预训练权重
net = SeqAttnPredictor(SEQ_LEN)
weight_file = './results/seq_attn_predictor.pth'
net.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
net.eval()


def predict(test_x):
    """
    股价预测接口
    :param test_x: ndarray (n, 14)，连续 14 个交易日收盘价
    :return: ndarray (n, 1)，第 15 个交易日的预测收盘价
    """
    n = test_x.shape[0]

    # 归一化到 [0, 1]（基于固定价格区间）
    normalizer = MinMaxScaler().fit(np.array([0, 300]).reshape(-1, 1))
    scaled = normalizer.transform(test_x.reshape(-1, 1)).reshape(n, 14)

    # 计算日间差分作为模型输入
    delta_input = np.diff(scaled, axis=1)

    # 前向推理
    with torch.no_grad():
        inp = torch.tensor(delta_input, dtype=torch.float32)
        delta_hat = net(inp).cpu().numpy().flatten()

    # 由最后一天的归一化价格加上预测增量，再逆变换回真实价格
    last_scaled = scaled[:, -1]
    pred_scaled = last_scaled + delta_hat
    result = normalizer.inverse_transform(pred_scaled.reshape(-1, 1))

    assert isinstance(result, np.ndarray)
    assert result.shape == (n, 1)
    return result
