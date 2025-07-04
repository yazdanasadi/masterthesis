# run_plot.py
import os
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.fld_icc import FLD
from model_ablation import SimpleFLD
from train_fld import SyntheticICUDataset, collate_fn, compute_train_stats

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

def train_with_log(model, train_loader, val_loader, epochs=100, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    nmse_log = []

    for epoch in range(1, epochs + 1):
        for T, X, M, TO, Y, MO in train_loader:
            T, X, M, TO, Y, MO = [t.to(DEVICE) for t in (T, X, M, TO, Y, MO)]
            Y_hat = model(T, X, M, TO)
            loss = NMSE(Y, Y_hat, MO)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_nmse = evaluate(model, val_loader)
        nmse_log.append(val_nmse)
        print(f"Epoch {epoch}: Val NMSE = {val_nmse:.4f}")

    return nmse_log

def plot_nmse_curve(logs, labels, filename=" nmse_epochs.png "):
    plt.figure()
    for log, label in zip(logs, labels):
        plt.plot(log, label=label)
    plt.xlabel(" Epoch ")
    plt.ylabel(" Validation NMSE ")
    plt.title(" Epochs vs NMSE ")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f" [✔] Saved NMSE plot: {filename} ")

def plot_attention_heatmap(attn_weights, filename="attention.png"):
    attn_mean = attn_weights.mean(dim=(0, 1)).detach().cpu().numpy()  # [P, S]
    plt.figure(figsize=(10, 4))
    plt.imshow(attn_mean, aspect='auto', cmap='viridis')
    plt.xlabel(" Time × Channel (S) ")
    plt.ylabel(" Basis Function (P) ")
    plt.title(" Average Attention Weights ")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[✔] Saved attention heatmap: {filename}")

if __name__ == "__main__":
    # Load data
    data_dir = "experiments/syntheticdata"
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")])
    random.shuffle(files)
    train_f, val_f = files[:80], files[80:90]
    mean_t, std_t = compute_train_stats(train_f)

    train_ds = SyntheticICUDataset(train_f, obs_frac=0.7, mean=mean_t, std=std_t)
    val_ds = SyntheticICUDataset(val_f, obs_frac=0.7, mean=mean_t, std=std_t)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # FLD (with attention)
    print(" Training FLD (with attention)...")
    model_attn = FLD(input_dim=3, latent_dim=32, num_heads=4, embed_dim=64, function="L").to(DEVICE)
    log_attn = train_with_log(model_attn, train_loader, val_loader, epochs=30)

    # SimpleFLD (no attention)
    print(" Training SimpleFLD (no attention)... ")
    model_simple = SimpleFLD(input_dim=3, latent_dim=32, embed_dim=64, function="L").to(DEVICE)
    log_simple = train_with_log(model_simple, train_loader, val_loader, epochs=30)

    # Plot NMSE curves
    plot_nmse_curve(
        logs=[log_attn, log_simple],
        labels=["FLD with Attention", "SimpleFLD (No Attention)"]
    )

    # Extract and plot attention
    print(" Extracting attention weights...")
    model_attn.eval()
    with torch.no_grad():
        for T, X, M, TO, Y, MO in val_loader:
            T, X, M, TO, Y, MO = [t.to(DEVICE) for t in (T, X, M, TO, Y, MO)]
            _, attn_weights = model_attn.attn(
                Q=torch.randn(X.shape[0], model_attn.P, model_attn.E).to(DEVICE),
                K=model_attn.channel_embed.unsqueeze(0).unsqueeze(0).expand(X.shape[0], X.shape[1]*X.shape[2], -1),
                V=model_attn.channel_embed.unsqueeze(0).unsqueeze(0).expand(X.shape[0], X.shape[1]*X.shape[2], -1),
                mask=torch.ones(X.shape[0], X.shape[1]*X.shape[2], dtype=torch.bool).to(DEVICE),
                return_attn=True
            )
            plot_attention_heatmap(attn_weights)
            break  # only plot one batch
