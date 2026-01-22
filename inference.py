
import torch
from models.guided_unet import GuidedUNet

def enhance(section, model_path):
    model = GuidedUNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        return model(section)
