
import argparse
import glob
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import jit
from torch.nn.utils.rnn import pad_sequence as pad
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm

import numpy as _np
_np.float128   = getattr(_np, "float128",   _np.float64)
_np.complex256 = getattr(_np, "complex256", _np.complex128)

from models.fld_cc_cross import FLD  

class GoodwinDataset(Dataset):
    def __init__(self, files, sparsity=0.9, fold=0, obs_frac=0.7):
        if not files:
            raise FileNotFoundError("No parquet files passed to GoodwinDataset!")
        torch.manual_seed(fold)
        dfs = [pd.read_parquet(f) for f in files]
        T_max = max(df["t"].max() for df in dfs)
        cols  = [c for c in dfs[0].columns if c != "t"]

        T_in_list, X_in_list, M_in_list = [], [], []
        T_out_list, Y_out_list, M_out_list = [], [], []

        # sequences
        for df in dfs:
            t_np     = df["t"].values
            obs_cut  = obs_frac * T_max
            mask_in  = t_np <= obs_cut
            mask_out = t_np > obs_cut

            # T and X_in
            T_in_list.append(torch.tensor(t_np, dtype=torch.float32) / T_max)
            X_raw = torch.tensor(df[cols].values, dtype=torch.float32)
            X_in_list.append(X_raw)
            # T_out and Y_out
            T_out_list.append(torch.tensor(t_np, dtype=torch.float32) / T_max)
            Y_raw = torch.tensor(df[cols].values, dtype=torch.float32)
            Y_out_list.append(Y_raw)

            # Masks for sparsity and forecasting window
            M_in_list.append((torch.rand_like(X_raw) > sparsity) & torch.tensor(mask_in)[:,None])
            M_out_list.append((torch.rand_like(Y_raw) > sparsity) & torch.tensor(mask_out)[:,None])

        # normalization: X_in and Y_out (using stats from X_in)
        all_X = torch.cat([Xi for Xi in X_in_list], dim=0)
        mean  = all_X.mean(0)
        std   = all_X.std(0).clamp(min=1e-6)
        X_in_list  = [(Xi - mean) / std for Xi in X_in_list]
        Y_out_list = [(Yi - mean) / std for Yi in Y_out_list]

        # padding: sequences must share the same time dimension
        # (pad T_in, X_in, M_in to same T length)
        self.T_in  = pad(T_in_list, batch_first=True)  # (B, T_max_len)
        self.X_in  = pad(X_in_list, batch_first=True, padding_value=float("nan"))  # (B, T, C)
        self.M_in  = pad(M_in_list, batch_first=True)  # (B, T, C)

        self.T_out = pad(T_out_list, batch_first=True)
        self.Y_out = pad(Y_out_list, batch_first=True, padding_value=float("nan"))
        self.M_out = pad(M_out_list, batch_first=True)

    def __len__(self):
        return self.X_in.size(0)

    def __getitem__(self, idx):
        return (
            self.T_in[idx], self.X_in[idx], self.M_in[idx],
            self.T_out[idx], self.Y_out[idx], self.M_out[idx],
        )

def get_loaders(path, batch_size, sparsity, fold, obs_frac):
    base      = os.path.dirname(os.path.abspath(__file__))
    full_glob = os.path.join(base, path, "*.parquet")
    print(f"Looking for .parquet with: {full_glob}")
    files = glob.glob(full_glob)
    if not files:
        sys.exit(f"ERROR: no .parquet files found under `{full_glob}`")
    random.seed(fold)
    random.shuffle(files)
    n = len(files)
    train_files = files[:int(0.7*n)]
    val_files   = files[int(0.7*n):int(0.8*n)]
    test_files  = files[int(0.8*n):]

    return (
        DataLoader(GoodwinDataset(train_files, sparsity, fold, obs_frac),
                   batch_size=batch_size, shuffle=True),
        DataLoader(GoodwinDataset(val_files,   sparsity, fold, obs_frac),
                   batch_size=batch_size, shuffle=False),
        DataLoader(GoodwinDataset(test_files,  sparsity, fold, obs_frac),
                   batch_size=batch_size, shuffle=False),
    )

def main():
    parser = argparse.ArgumentParser(description="Train FLD on Goodwin data")
    parser.add_argument("--path",        default="experiments/Goodwin_data",
                        help="folder containing .parquet files")
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--sparsity",    type=float, default=0.9)
    parser.add_argument("--fold",        type=int,   default=0)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--latent-dim",  type=int,   default=20)
    parser.add_argument("--emb-dim",     type=int,   default=4)
    parser.add_argument("--heads",       type=int,   default=2)
    parser.add_argument("--depth",       type=int,   default=1)
    parser.add_argument("--function",    choices=("C","L","Q","S"), default="S")
    parser.add_argument("--residual-cycle", action="store_true")
    parser.add_argument("--cycle-length",  type=int, default=24)
    parser.add_argument("--obs-frac",      dest="obs_frac", type=float, default=0.7,
                        help="fraction of each series used for input")
    
    parser.add_argument(
    "--cross-channel",
    dest="cross_channel",
    action="store_true",
    help="enable cross-channel attention"  
    )
    parser.add_argument(
        "--no-cross-channel",
        dest="cross_channel",
        action="store_false",
        help="disable cross-channel attention"
    )
    parser.set_defaults(cross_channel=True)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_loaders(
        args.path, args.batch_size, args.sparsity, args.fold, args.obs_frac
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # infer channels
    _, X0, M0, _, Y0, MY0 = next(iter(train_loader))
    C = X0.size(-1)

    model = FLD(
        input_dim=C,
        latent_dim=args.latent_dim,
        embed_dim_per_head=args.emb_dim,
        num_heads=args.heads,
        function=args.function,
        depth=args.depth,
        device=device,
        residual_cycle=args.residual_cycle,
        cycle_length=args.cycle_length,
        cross_channel=args.cross_channel
    ).to(device)

    def MSE(y, yhat, mask): return torch.sum(mask*(y-yhat)**2)/torch.sum(mask)
    loss_fn = jit.script(MSE)
    opt     = AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            T, X, M, TY, Y, MY = [t.to(device) for t in batch]
            Yhat = model(T, X, M, TY)
            loss = loss_fn(Y, Yhat, MY)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())

        model.eval()
        vsum, vcnt = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                T, X, M, TY, Y, MY = [t.to(device) for t in batch]
                Yhat = model(T, X, M, TY)
                l = loss_fn(Y, Yhat, MY) * MY.sum()
                vsum += l.item(); vcnt += MY.sum().item()
        if vcnt == 0:
            print("Warning: no valid targets in validation; skipping loss.")
            vloss = float("inf")
        else:
            vloss = vsum / vcnt
        print(f"Epoch {epoch} | Train={np.mean(train_losses):.4f} Val={vloss:.4f}")

        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), "best_goodwin_fld.pt")

    # Testing
    model.load_state_dict(torch.load("best_goodwin_fld.pt"))
    model.eval()
    tsum, tcnt = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            T, X, M, TY, Y, MY = [t.to(device) for t in batch]
            Yhat = model(T, X, M, TY)
            l = loss_fn(Y, Yhat, MY) * MY.sum()
            tsum += l.item(); tcnt += MY.sum().item()
    if tcnt == 0:
        print("Warning: no valid targets in test; test MSE unavailable.")
    else:
        print("Test MSE:", tsum / tcnt)

if __name__ == "__main__":
    main()