#!/usr/bin/env python
import os
import itertools
import random
import argparse
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from models.fld_icc import FLD
from train_fld import SyntheticICUDataset, compute_train_stats, collate_fn

class NoisyDataset(Dataset):
    """ Wraps another Dataset and adds Gaussian noise to the inputs. """
    def __init__(self, base_ds: Dataset, noise_std: float = 0.01):
        self.base_ds = base_ds
        self.noise_std = noise_std
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        Tin, Xin, Min, Tout, Yout, Mout = self.base_ds[idx]
        Xin = Xin + torch.randn_like(Xin) * self.noise_std
        return Tin, Xin, Min, Tout, Yout, Mout


def hyperparam_search(
    data_dir: str,
    out_csv: str = "hp_results.csv",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda",
    add_noise: bool = False,
    noise_std: float = 0.01
):
    # Prepare data splits and compute normalization stats
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")])
    random.shuffle(files)
    n = len(files)
    train_f, val_f, _ = files[:int(0.8*n)], files[int(0.8*n):int(0.9*n)], files[int(0.9*n):]
    mean_t, std_t = compute_train_stats(train_f)
    # Build full grid of hyperparameters
    grid = []
    for hd, heads, depth, head_dim, obs in itertools.product(
            [32, 128, 256, 512], [4, 8], [2, 4], [2, 8], [0.5, 0.7, 0.8]):
        grid.append({
            "latent_dim":    hd,
            "heads":         heads,
            "decoder_depth": depth,
            "head_dim":      head_dim,
            "embed_dim":     heads * head_dim,
            "obs_frac":      obs,
            "noise":         add_noise
        })
    # Load existing results to skip completed configs
    done = set()
    if os.path.exists(out_csv):
        df_done = pd.read_csv(out_csv)
        for _, row in df_done.iterrows():
            done.add((
                int(row.latent_dim),
                int(row.heads),
                int(row.decoder_depth),
                int(row.head_dim),
                float(row.obs_frac),
                bool(row.noise)
            ))
    # Filter grid to only untested configs
    to_run = [cfg for cfg in grid if (
        cfg["latent_dim"], cfg["heads"], cfg["decoder_depth"],
        cfg["head_dim"], cfg["obs_frac"], cfg["noise"]
    ) not in done]
    if not to_run:
        print("All hyperparameter combinations have been tested. Nothing to run.")
        return
    # Open CSV for append
    write_header = not os.path.exists(out_csv)
    for cfg in to_run:
        print("Running config:", cfg)
        # Prepare datasets
        train_ds = SyntheticICUDataset(train_f, cfg["obs_frac"], mean_t, std_t)
        val_ds   = SyntheticICUDataset(val_f,   cfg["obs_frac"], mean_t, std_t)
        if add_noise:
            train_ds = NoisyDataset(train_ds, noise_std)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        # Build model
        model = FLD(
            input_dim=3,
            latent_dim=cfg["latent_dim"],
            num_heads=cfg["heads"],
            embed_dim=cfg["embed_dim"],
            function="L",
            residual_cycle=False,
            cycle_length=24
        ).to(device)
        # Decoder
        layers = []
        for _ in range(cfg["decoder_depth"]):
            layers += [torch.nn.Linear(cfg["latent_dim"], cfg["latent_dim"]), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(cfg["latent_dim"], 3))
        model.out = torch.nn.Sequential(*layers).to(device)
        # Loss and optimizer
        def NMSE(y, yhat, mask):
            num = torch.sum(mask * (y - yhat)**2)
            den = torch.sum(mask * y**2) + 1e-6
            return num / den
        loss_fn   = torch.jit.script(NMSE)
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-5
        )
        # Train and validate
        best_val = float("inf")
        for epoch in range(1, epochs+1):
            model.train()
            for T, X, M, TO, Y, MO in train_loader:
                T, X, M, TO, Y, MO = [t.to(device) for t in (T, X, M, TO, Y, MO)]
                yhat = model(T, X, M, TO)
                loss = loss_fn(Y, yhat, MO)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            v_sum, v_cnt = 0.0, 0
            with torch.no_grad():
                for T, X, M, TO, Y, MO in val_loader:
                    T, X, M, TO, Y, MO = [t.to(device) for t in (T, X, M, TO, Y, MO)]
                    yhat = model(T, X, M, TO)
                    l = loss_fn(Y, yhat, MO) * MO.sum()
                    v_sum += l.item()
                    v_cnt += MO.sum().item()
            val_nmse = v_sum / (v_cnt + 1e-6)
            scheduler.step(val_nmse)
            best_val = min(best_val, val_nmse)
        # Append result immediately
        cfg["best_val_nmse"] = best_val
        pd.DataFrame([cfg]).to_csv(out_csv, mode="a", header=write_header, index=False)
        write_header = False
    
    print(f"Hyperparameter search complete; results saved to {out_csv}")
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="FLD hyperparameter search")
    parser.add_argument("--data-dir",   type=str,   default="experiments/syntheticdata")
    parser.add_argument("--out-csv",    type=str,   default="hp_results.csv")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--add-noise",  action="store_true")
    parser.add_argument("--noise-std",  type=float, default=0.01)
    args = parser.parse_args()

    print("Starting hyperparameter search with args:", args)
    hyperparam_search(

        data_dir   = args.data_dir,
        out_csv    = args.out_csv,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        device     = args.device,
        add_noise  = args.add_noise,
        noise_std  = args.noise_std

    )
