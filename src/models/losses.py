import torch

def reconstruction_loss(pred, target):
    return torch.mean(torch.abs(pred - target))
