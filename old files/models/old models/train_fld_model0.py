#!/usr/bin/env python
import argparse
import numpy as np
import torch
from torch.optim import AdamW

from models.fld_model import FLD
from data.synthetic_dataset import get_synthetic_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Train FLD model")
    return parser.parse_args()

def mse_loss(y, yhat, mask):
    return torch.sum(mask * (y - yhat)**2) / torch.sum(mask)

def main():
    loader = get_synthetic_loader(n_samples=200, obs_len=24, pred_len=24, input_dim=96, batch_size=16, seed=0)
    device = torch.device("cpu")
    model = FLD(input_dim=96, latent_dim=20, embed_dim_per_head=4, num_heads=2, function="C", device=device).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1, 6):
        losses = []
        model.train()
        for T, X, M, TY, Y, MY in loader:
            optimizer.zero_grad()
            Yh = model(T, X, M, TY)
            loss = mse_loss(Y, Yh, MY)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch}/5 - Loss: {np.mean(losses):.4f}")

if __name__ == "__main__":
    main()
