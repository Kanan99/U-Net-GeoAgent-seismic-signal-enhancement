
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.guided_unet import GuidedUNet
from models.evaluator import SignalEvaluator
from losses.signal_loss import l1_loss
from losses.spectral_loss import spectral_consistency
from config import Config

def train_loop(loader):
    enhancer = GuidedUNet().to(Config.device)
    evaluator = SignalEvaluator().to(Config.device)

    opt_e = torch.optim.Adam(enhancer.parameters(), lr=Config.lr_primary)
    opt_v = torch.optim.Adam(evaluator.parameters(), lr=Config.lr_evaluator)

    for epoch in range(Config.num_epochs):
        for degraded, clean in tqdm(loader):
            degraded, clean = degraded.to(Config.device), clean.to(Config.device)

            enhanced = enhancer(degraded)

            loss_signal = l1_loss(enhanced, clean)
            loss_spec = spectral_consistency(enhanced, clean)

            score_real = evaluator(clean)
            score_fake = evaluator(enhanced.detach())

            loss_eval = -(score_real.mean() - score_fake.mean())

            opt_v.zero_grad()
            loss_eval.backward()
            opt_v.step()

            total_loss = loss_signal + 0.1 * loss_spec - score_fake.mean()

            opt_e.zero_grad()
            total_loss.backward()
            opt_e.step()

        print(f"Epoch {epoch+1} | Enhancement Loss: {total_loss.item():.4f}")
