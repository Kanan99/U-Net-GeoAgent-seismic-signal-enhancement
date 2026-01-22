
import torch

def spectral_consistency(pred, target):
    pf = torch.fft.fft2(pred)
    tf = torch.fft.fft2(target)
    return torch.mean(torch.abs(torch.abs(pf) - torch.abs(tf)))
