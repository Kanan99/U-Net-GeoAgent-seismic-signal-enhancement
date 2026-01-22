
import torch

def snr(clean, enhanced):
    noise = clean - enhanced
    return 10 * torch.log10(torch.mean(clean**2) / torch.mean(noise**2))
