
import torch
import torch.nn as nn

class SignalEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x).mean(dim=[1,2,3])
