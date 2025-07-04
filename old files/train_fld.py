#!/usr/bin/env python
# coding: utf-8

import argparse, os, glob, random
import numpy as np
import pandas as pd
import torch
from torch import jit
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.fld_icc import FLD 

class SyntheticICUDataset(Dataset):
    def __init__(self, files, obs_frac=0.7, mean=None, std=None):
        self.files    = files
        self.obs_frac = obs_frac
        self.mean     = mean
        self.std      = std

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        # Load dataframe
        df = pd.read_parquet(self.files[idx])
        t_np = df['t'].to_numpy()
        t    = torch.from_numpy(t_np).float()  # [T]

        # Protect against zero max time
        t_max = t.max().item()
        if t_max <= 0:
            t_max = 1e-6
            print(f"[Dataset __getitem__] idx={idx} warning: t.max()<=0, using t_max={t_max}")
        t_norm = t / t_max  # [T]

        # Load features and mask
        arr = df[['heart_rate','blood_pressure','respiratory_rate']].to_numpy()
        X_full = torch.from_numpy(arr).float()  # [T, C]
        M_full = ~torch.isnan(X_full)          # [T, C]
        X_full = torch.nan_to_num(X_full, nan=0.0)

        # Normalize features if stats provided
        if self.mean is not None:
            mean = self.mean.view(-1)  # (C,)
            std  = self.std.view(-1)
            X_full = (X_full - mean) / (std + 1e-6)

        # Create input/output masks by observation fraction
        cut      = self.obs_frac
        mask_in  = t_norm <= cut            # [T]
        mask_out = t_norm >  cut            # [T]

        # Ensure at least one timestep in each split
        if mask_in.sum().item() == 0:
            mask_in[0] = True
            mask_out[0] = False
            print(f"[Dataset __getitem__] idx={idx} warning: no mask_in True, forcing first timestep")
        if mask_out.sum().item() == 0:
            mask_out[-1] = True
            mask_in[-1] = False
            # print(f"[Dataset __getitem__] idx={idx} warning: no mask_out True, forcing last timestep")

        # DEBUG: print shapes
        # print(f"[Dataset __getitem__] idx={idx}, t_norm.shape={t_norm.shape}, X_full.shape={X_full.shape}, mask_in.sum()={mask_in.sum().item()}, mask_out.sum()={mask_out.sum().item()}")

        # Input sequences
        Tin  = t_norm[mask_in]              # [T_in]
        Xin  = X_full[mask_in, :]           # [T_in, C]
        Min  = M_full[mask_in, :]           # [T_in, C]

        # Output sequences
        Tout = t_norm[mask_out]             # [T_out]
        Yout = X_full[mask_out, :]          # [T_out, C]
        Mout = M_full[mask_out, :]          # [T_out, C]

        return Tin, Xin, Min, Tout, Yout, Mout


def collate_fn(batch):
    Ts, Xs, Ms, TOs, Ys, Os = zip(*batch)
    pad = torch.nn.utils.rnn.pad_sequence
    Ts  = pad(Ts,  batch_first=True, padding_value=0.0)
    Xs  = pad(Xs,  batch_first=True, padding_value=0.0)
    Ms  = pad(Ms,  batch_first=True, padding_value=False)
    TOs = pad(TOs, batch_first=True, padding_value=0.0)
    Ys  = pad(Ys,  batch_first=True, padding_value=0.0)
    Os  = pad(Os,  batch_first=True, padding_value=False)
    return Ts, Xs, Ms, TOs, Ys, Os


def compute_train_stats(files):
    """
    Compute per-feature mean and std, ignoring NaN values across all training files.
    """
    arrs = []
    for f in files:
        df = pd.read_parquet(f)
        arr = df[['heart_rate','blood_pressure','respiratory_rate']].to_numpy()
        arrs.append(arr)
    X = np.vstack(arrs)  # shape [N, C], may contain NaNs
    # Compute nan-aware statistics
    mean = np.nanmean(X, axis=0)  # (C,)
    std  = np.nanstd(X, axis=0)   # (C,)
    # Handle any features with zero std
    std[std == 0] = 1.0
    # Convert to tensors
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t  = torch.tensor(std,  dtype=torch.float32)
    return mean_t, std_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',      type=str,   default='experiments/syntheticdata')
    parser.add_argument('--batch-size',    type=int,   default=16)
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--latent-dim',    type=int,   default=20)
    parser.add_argument('--embed-dim',     type=int,   default=64)
    parser.add_argument('--heads',         type=int,   default=4)
    parser.add_argument('--function',      choices=('C','L','Q','S'), default='L')
    parser.add_argument('--residual-cycle',action='store_true')
    parser.add_argument('--cycle-length',  type=int,   default=24)
    parser.add_argument('--obs-frac',      type=float, default=0.7)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data_dir, '*.parquet'))
    random.shuffle(files)
    n = len(files)
    train_f, val_f, test_f = files[:int(0.8*n)], files[int(0.8*n):int(0.9*n)], files[int(0.9*n):]

    mean_t, std_t = compute_train_stats(train_f)

    train_ds = SyntheticICUDataset(train_f, args.obs_frac, mean_t, std_t)
    val_ds   = SyntheticICUDataset(val_f,   args.obs_frac, mean_t, std_t)
    test_ds  = SyntheticICUDataset(test_f,  args.obs_frac, mean_t, std_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = FLD(
        input_dim=3,
        latent_dim=args.latent_dim,
        num_heads=args.heads,
        embed_dim=args.embed_dim,
        function=args.function,
        residual_cycle=args.residual_cycle,
        cycle_length=args.cycle_length
    ).to(device)

    # richer decoder
    hidden = args.latent_dim * 2
    model.out = torch.nn.Sequential(
        torch.nn.Linear(args.latent_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 3),
    ).to(device)

    # Normalized MSE: sum(mask*(y-ŷ)^2) / sum(mask*y^2)
    def NMSE(y, yhat, mask):
        num = torch.sum(mask * (y - yhat)**2)
        den = torch.sum(mask * (y       )**2) + 1e-6
        return num / den
    loss_fn   = jit.script(NMSE)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5
    )

    writer = SummaryWriter()
    best_val = float('inf')

    for epoch in range(1, args.epochs+1):
        model.train()
        train_losses = []
        for batch_idx, (T, X, M, TO, Y, MO) in enumerate(train_loader):
            # DEBUG: print batch shapes
            # print(f"[Train] epoch={epoch}, batch={batch_idx}, T={T.shape}, X={X.shape}, Y={Y.shape}, MO.sum={MO.sum().item()}")
            T, X, M, TO, Y, MO = [t.to(device) for t in (T,X,M,TO,Y,MO)]
            yhat = model(T, X, M, TO)
            # Compute loss
            loss = loss_fn(Y, yhat, MO)
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"[ERROR] NaN loss at epoch={epoch}, batch={batch_idx}")
                print(f"Y stats: min={Y.min().item()}, max={Y.max().item()}, mean={Y.mean().item()}")
                print(f"yhat stats: min={yhat.min().item()}, max={yhat.max().item()}, mean={yhat.mean().item()}")
                import sys; sys.exit(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_nmse = np.mean(train_losses)
        writer.add_scalar('NMSE/Train', train_nmse, epoch)

        model.eval()
        v_sum, v_cnt = 0.0, 0
        with torch.no_grad():
            for T, X, M, TO, Y, MO in val_loader:
                T, X, M, TO, Y, MO = [t.to(device) for t in (T,X,M,TO,Y,MO)]
                yhat = model(T, X, M, TO)
                l = loss_fn(Y, yhat, MO) * MO.sum()
                v_sum += l.item()
                v_cnt += MO.sum().item()
        val_nmse = v_sum / (v_cnt + 1e-6)
        writer.add_scalar('NMSE/Val', val_nmse, epoch)
        print(f"Epoch {epoch} | Train NMSE={train_nmse:.4f} | Val NMSE={val_nmse:.4f}")

        scheduler.step(val_nmse)
        if val_nmse < best_val:
            best_val = val_nmse
            torch.save(model.state_dict(), 'best_synth_fld_nmse.pt')

    # Testing
    model.load_state_dict(torch.load('best_synth_fld_nmse.pt'))
    model.eval()
    t_sum, t_cnt = 0.0, 0
    with torch.no_grad():
        for T, X, M, TO, Y, MO in test_loader:
            T, X, M, TO, Y, MO = [t.to(device) for t in (T,X,M,TO,Y,MO)]
            yhat = model(T, X, M, TO)
            l = loss_fn(Y, yhat, MO) * MO.sum()
            t_sum += l.item()
            t_cnt += MO.sum().item()
    test_nmse = t_sum / (t_cnt + 1e-6)
    writer.add_scalar('NMSE/Test', test_nmse)
    print(f"Test NMSE: {test_nmse:.4f}")

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
