# ==================== 模型预测代码 ====================
import numpy as np
import torch
import torch.nn as nn

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

model = MLPNet()
model_path = 'results/mymodel.pt'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

def predict(test_x):
    '''
    :param test_x: numpy array, shape (n, 14)
    :return: numpy array, shape (n,)
    '''
    base = test_x[:, -1:]
    x_norm = test_x / base
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    with torch.no_grad():
        y_norm = model(x_tensor)
    y_pred = y_norm.numpy() * base
    return y_pred.reshape(-1)
