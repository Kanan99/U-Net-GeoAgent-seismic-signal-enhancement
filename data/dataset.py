
import torch
from torch.utils.data import Dataset
import numpy as np

class SeismicDataset(Dataset):
    def __init__(self, clean_sections, degraded_sections):
        self.clean = clean_sections
        self.degraded = degraded_sections

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.degraded[idx], dtype=torch.float32),
            torch.tensor(self.clean[idx], dtype=torch.float32),
        )
