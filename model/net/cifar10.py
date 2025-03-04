import torch.nn as nn

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # 输入通道3（RGB）
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),      # 32x32输入经过3次MaxPool变为4x4
            nn.Linear(64, 10)           # CIFAR-10有10类
        )

    def forward(self, x):
        return self.model(x)