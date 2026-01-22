
import torch

def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))
