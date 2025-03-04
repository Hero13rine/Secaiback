import torch
import torch.nn as nn

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3),  # 1x28x28 -> 32x26x26
            nn.ReLU(),
            nn.MaxPool2d(2),      # 32x13x13
            nn.Conv2d(32, 64, 3), # 64x11x11
            nn.ReLU(),
            nn.MaxPool2d(2)       # 64x5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)