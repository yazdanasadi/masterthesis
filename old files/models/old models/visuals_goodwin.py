#!/usr/bin/env python
# coding: utf-8

import os, glob
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Import your modified FLD (with residual_cycle & cross-channel off/on via flag)
from models.fld_cc_cross import FLD

class GoodwinDataset(Dataset):
    """Splits each series into an observed portion and a forecast portion."""
    def __init__(self, path, sparsity=0.9, fold=0, obs_frac=0.7):
        torch.manual_seed(fold)
        files = glob.glob(os.path.join(path, "*.parquet"))
        dfs = [pd.read_parquet(f) for f in files]
        T_max = max(df["t"].max() for df in dfs)
        cols = [c for c in dfs[0].columns if c != "t"]

        self.T_in, self.X_in, self.M_in = [], [], []
        self.T_out, self.Y_out, self.M_out = [], [], []

        for df in dfs:
            t = torch.tensor(df["t"].values, dtype=torch.float32)
            X = torch.tensor(df[cols].values, dtype=torch.float32)

            obs_cut = obs_frac * T_max
            mask_in  = (t <= obs_cut)
            mask_out = (t >  obs_cut)

            # observed
            Ti = t[mask_in]  / T_max
            Xi = X[mask_in]
            Mi = (torch.rand_like(Xi) > sparsity)

            # forecast
            To = t[mask_out] / T_max
            Yo = X[mask_out]
            Mo = (torch.rand_like(Yo) > sparsity)

            # normalize Xi & Yo using Xi stats
            mean = Xi.mean(0)
            std  = Xi.std(0).clamp(min=1e-6)
            Xi = (Xi - mean) / std
            Yo = (Yo - mean) / std

            self.T_in.append(Ti)
            self.X_in.append(Xi)
            self.M_in.append(Mi)

            self.T_out.append(To)
            self.Y_out.append(Yo)
            self.M_out.append(Mo)

        # pad to max length per split
        self.T_in  = pad(self.T_in,  batch_first=True)  # (B, Tin)
        self.X_in  = pad(self.X_in,  batch_first=True, padding_value=float("nan"))
        self.M_in  = pad(self.M_in,  batch_first=True)

        self.T_out = pad(self.T_out, batch_first=True)  # (B, Tout)
        self.Y_out = pad(self.Y_out, batch_first=True, padding_value=float("nan"))
        self.M_out = pad(self.M_out, batch_first=True)

    def __len__(self): return self.X_in.size(0)
    def __getitem__(self, idx):
        return (
            self.T_in[idx], self.X_in[idx], self.M_in[idx],
            self.T_out[idx], self.Y_out[idx], self.M_out[idx],
        )

def visualize_goodwin(
    data_path="experiments/Goodwin_data",
    ckpt="best_goodwin_fld.pt",
    sample_idx=0,
    obs_frac=0.7,
    sparsity=0.9,
):
    device = torch.device("cpu")

    # load one sample
    ds = GoodwinDataset(data_path, sparsity, fold=0, obs_frac=obs_frac)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    T_in, X_in, M_in, T_out, Y_out, M_out = next(iter(loader))
    T_in, X_in, M_in = T_in.to(device),  X_in.to(device),  M_in.to(device)
    T_out, Y_out, M_out = T_out.to(device), Y_out.to(device), M_out.to(device)

    B, Tin, C = X_in.shape
    Tout      = T_out.shape[1]
    print(f"Sample shapes: X_in=(B={B},Tin={Tin},C={C}), Y_out=(B,{Tout},{C})")

    # build model
    model = FLD(
        input_dim=C,
        latent_dim=20,
        embed_dim_per_head=8,
        num_heads=2,
        function="L",
        depth=1,
        device=device,
        residual_cycle=True,
        cycle_length=24,
    ).to(device)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        # compute cycle baseline on observed
        c_in, c_base = model._get_cycle(T_in, X_in, M_in)
        # forecast
        Yhat = model(T_in, X_in, M_in, T_out)
        # cycle baseline on forecast
        idx = torch.floor(T_out * model.cycle_length).long() % model.cycle_length
        c_fut = c_base.gather(1, idx.unsqueeze(-1).expand(-1,-1,C))

    # to numpy
    Tin_n  = T_in[0].cpu().numpy()
    Xin_n  = X_in[0].cpu().numpy()
    Tout_n = T_out[0].cpu().numpy()
    Yout_n = Y_out[0].cpu().numpy()
    Yhat_n = Yhat[0].cpu().numpy()
    c_in_n  = c_in[0].cpu().numpy()
    c_fut_n = c_fut[0].cpu().numpy()

    # plot channels 0 & 1
    for ch in [0,1]:
        plt.figure(figsize=(6,3))
        # observed
        plt.plot(Tin_n, Xin_n[:,ch], label="Observed True")
        plt.plot(Tin_n, c_in_n[:,ch], '--', label="Baseline In")
        # forecast window
        plt.plot(Tout_n, Yout_n[:,ch], label="True Forecast")
        plt.plot(Tout_n, Yhat_n[:,ch], '--', label="FLD Pred")
        plt.plot(Tout_n, c_fut_n[:,ch], ':', label="Baseline Out")
        plt.title(f"Goodwin Channel {ch}")
        plt.xlabel("Normalized Time")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_goodwin()
