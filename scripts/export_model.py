import torch

def export(model, path):
    torch.save(model.state_dict(), path)
