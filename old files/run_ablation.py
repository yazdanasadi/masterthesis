# run_ablation.py
import torch
from torch.utils.data import DataLoader
from models.fld_icc import FLD
from model_ablation import SimpleFLD
from train_fld import SyntheticICUDataset, collate_fn, compute_train_stats
import os
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def NMSE(y, yhat, mask):
    num = torch.sum(mask * (y - yhat) ** 2)
    den = torch.sum(mask * y ** 2) + 1e-6
    return num / den

def evaluate(model, loader):
    model.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for T, X, M, TO, Y, MO in loader:
            T, X, M, TO, Y, MO = [t.to(DEVICE) for t in (T, X, M, TO, Y, MO)]
            Y_hat = model(T, X, M, TO)
            loss = NMSE(Y, Y_hat, MO) * MO.sum()
            total_loss += loss.item()
            total_count += MO.sum().item()
    return total_loss / (total_count + 1e-6)

def train_one(model, loader, epochs=5, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = NMSE

    for epoch in range(epochs):
        for T, X, M, TO, Y, MO in loader:
            T, X, M, TO, Y, MO = [t.to(DEVICE) for t in (T, X, M, TO, Y, MO)]
            Y_hat = model(T, X, M, TO)
            loss = loss_fn(Y, Y_hat, MO)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    data_dir = "experiments/syntheticdata"
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")])
    random.shuffle(files)
    train_f, val_f = files[:80], files[80:90]
    mean_t, std_t = compute_train_stats(train_f)

    train_ds = SyntheticICUDataset(train_f, obs_frac=0.7, mean=mean_t, std=std_t)
    val_ds = SyntheticICUDataset(val_f, obs_frac=0.7, mean=mean_t, std=std_t)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    print("Training FLD (with attention)...")
    model_attn = FLD(input_dim=3, latent_dim=32, num_heads=4, embed_dim=64, function="L").to(DEVICE)
    train_one(model_attn, train_loader)
    val_nmse_attn = evaluate(model_attn, val_loader)

    print("Training SimpleFLD (no attention)...")
    model_simple = SimpleFLD(input_dim=3, latent_dim=32, embed_dim=64, function="L").to(DEVICE)
    train_one(model_simple, train_loader)
    val_nmse_simple = evaluate(model_simple, val_loader)

    print(f"\n Validation NMSE")
    print(f"With Attention   : {val_nmse_attn:.4f}")
    print(f"Without Attention: {val_nmse_simple:.4f}")
