#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import os
import random
import time
import warnings
import logging
import yaml
import numpy as np
import torch
from torch import Tensor, jit
from torch.optim import AdamW
from datetime import datetime
from pathlib import Path
from random import SystemRandom

# add project root if necessary
sys.path.insert(0, "../")

# ----------------------------------------
# Command‐line arguments
# ----------------------------------------
parser = argparse.ArgumentParser(description="Training Script for FLD model.")
parser.add_argument("-c", "--config",
                    default=None, nargs=2, type=str,
                    help="load external config (file, id)")                     # + re-added config arg
parser.add_argument("-e",  "--epochs",       default=300,    type=int,   help="maximum epochs")
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")
parser.add_argument("-es", "--early-stop",   default=30,     type=int,   help="early stopping patience")
parser.add_argument("-bs", "--batch-size",   default=64,     type=int,   help="batch size")
parser.add_argument("-lr", "--learn-rate",   default=1e-4,   type=float, help="learning rate")
parser.add_argument("-wd", "--weight-decay", default=1e-3,   type=float, help="weight decay")
parser.add_argument("-ls", "--latent-dim",   default=20,     type=int,   help="latent dimension")
parser.add_argument("-hd", "--hidden-dim",   default=None,   type=int,   help="hidden layer size in decoder")
parser.add_argument("-ed", "--embedding-dim",default=4,      type=int,   help="per-head embedding dim")
parser.add_argument("-nh", "--num-heads",    default=2,      type=int,   help="number of attention heads")
parser.add_argument("-dp", "--depth",        default=1,      type=int,   help="depth of decoder MLP")
parser.add_argument("-fn", "--function",     default="C",    choices=("C","L","Q","S"), help="hidden function type")
parser.add_argument("-dset","--dataset",     default="ushcn", type=str,  help="dataset name")
parser.add_argument("-ot","--observation-time", default=24,  type=int,   help="observation window (hours)")
parser.add_argument("-fh","--forecast-horizon", default=24,  type=int,   help="forecast horizon (hours)")
parser.add_argument("-s", "--seed",         default=0,      type=int,   help="random seed")
# + NEW: Residual Cycle Forecasting flags
parser.add_argument("--residual-cycle", action="store_true",
                    help="enable Residual Cycle Forecasting (RCF)")
parser.add_argument("--cycle-length",    type=int, default=24,
                    help="cycle length (number of slots) for RCF")

ARGS = parser.parse_args()

# ----------------------------------------
# Optional external config loading
# ----------------------------------------
if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    vars(ARGS).update(**cfg[int(cfg_id)])

# ----------------------------------------
# Environment & reproducibility
# ----------------------------------------
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
import numpy as _np             # NEW
_np.float128   = getattr(_np, "float128",   _np.float64)    # NEW
_np.complex256 = getattr(_np, "complex256", _np.complex128)  # NEW
if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------------------
# Data & task selection
# ----------------------------------------
from tsdm.tasks.mimic_iii_debrouwer2019 import mimic_collate as task_collate_fn

if ARGS.dataset == "mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019 as Task
    INPUT_DIM = 96
elif ARGS.dataset == "mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021 as Task
    INPUT_DIM = 102
elif ARGS.dataset == "p12":
    from tsdm.tasks.physionet2012 import Physionet2012 as Task
    INPUT_DIM = 37
else:  # ushcn
    from tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019 as Task
    INPUT_DIM = 5

TASK = Task(condition_time=ARGS.observation_time,
            forecast_horizon=ARGS.forecast_horizon)

dloader_cfg = {
    "batch_size":  ARGS.batch_size,
    "shuffle":     True,
    "drop_last":   True,
    "pin_memory":  True,
    "num_workers": 4,
    "collate_fn":  task_collate_fn,
}
eval_cfg = {**dloader_cfg, "shuffle": False, "drop_last": False, "num_workers": 0}

TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_cfg)
VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **eval_cfg)
TEST_LOADER  = TASK.get_dataloader((ARGS.fold, "test"),  **eval_cfg)

# ----------------------------------------
# Metrics & loss
# ----------------------------------------
def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    return torch.sum(mask * (y - yhat)**2) / torch.sum(mask)

METRICS = {"MSE": jit.script(MSE)}
LOSS    = jit.script(MSE)

# ----------------------------------------
# Model instantiation
# ----------------------------------------
from models.fld_cc import FLD

MODEL = FLD(
    input_dim=INPUT_DIM,
    latent_dim=ARGS.latent_dim,
    embed_dim_per_head=ARGS.embedding_dim,
    num_heads=ARGS.num_heads,
    function=ARGS.function,
    depth=ARGS.depth,
    device=DEVICE,
    # +: passing RCF flags into the model
    residual_cycle=ARGS.residual_cycle,
    cycle_length=ARGS.cycle_length,
).to(DEVICE)

# ----------------------------------------
# Training / evaluation routines
# ----------------------------------------
def predict_fn(model, batch):
    T, X, M, TY, Y, MY = (t.to(DEVICE) for t in batch)
    YHAT = model(T, X, M, TY)
    return Y, YHAT, MY

# quick sanity check
batch = next(iter(TRAIN_LOADER))
MODEL.zero_grad(set_to_none=True)
Y, YHAT, MASK = predict_fn(MODEL, batch)
loss0 = LOSS(Y, YHAT, MASK)
assert torch.isfinite(loss0).item(), "Model collapsed!"
loss0.backward()
MODEL.zero_grad(set_to_none=True)

# ----------------------------------------
# Optimizer & scheduler
# ----------------------------------------
OPTIMIZER = AdamW(MODEL.parameters(), lr=ARGS.learn_rate,
                  weight_decay=ARGS.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER, mode="min", patience=10, factor=0.5, min_lr=5e-4, verbose=True
)

# ----------------------------------------
# Main training loop
# ----------------------------------------
best_val = float("inf")
early_ctr = 0
times = []
print(f"# Parameters: {sum(p.numel() for p in MODEL.parameters() if p.requires_grad)}")

for epoch in range(1, ARGS.epochs + 1):
    MODEL.train()
    train_losses = []
    start = time.time()
    for batch in TRAIN_LOADER:
        OPTIMIZER.zero_grad()
        Y, YHAT, MASK = predict_fn(MODEL, batch)
        loss = LOSS(Y, YHAT, MASK)
        loss.backward()
        OPTIMIZER.step()
        train_losses.append(loss.item())
    times.append(time.time() - start)
    train_loss = np.mean(train_losses)

    MODEL.eval()
    val_sum = 0.0
    val_cnt = 0.0
    with torch.no_grad():
        for batch in VALID_LOADER:
            Y, YHAT, MASK = predict_fn(MODEL, batch)
            l = LOSS(Y, YHAT, MASK) * MASK.sum()
            val_sum += l.item()
            val_cnt += MASK.sum().item()
    val_loss = val_sum / val_cnt

    print(f"Epoch {epoch} | Train={train_loss:.4f} Val={val_loss:.4f} Time={times[-1]:.2f}s")

    if val_loss < best_val:
        best_val = val_loss
        early_ctr = 0
        torch.save({
            "args":        ARGS,
            "epoch":       epoch,
            "state_dict":  MODEL.state_dict(),
            "optimizer":   OPTIMIZER.state_dict(),
            "loss":        train_loss,
        }, f"saved_models/FLD-{ARGS.function}_{ARGS.dataset}_{epoch}.pt")
    else:
        early_ctr += 1

    if early_ctr >= ARGS.early_stop:
        print(f"Stopping early at epoch {epoch}")
        break

    scheduler.step(val_loss)

# ----------------------------------------
# Final test pass
# ----------------------------------------
chkp = torch.load(f"saved_models/FLD-{ARGS.function}_{ARGS.dataset}_{epoch}.pt")
MODEL.load_state_dict(chkp["state_dict"])
MODEL.eval()
test_sum = 0.0
test_cnt = 0.0
with torch.no_grad():
    start_inf = time.time()
    for batch in TEST_LOADER:
        Y, YHAT, MASK = predict_fn(MODEL, batch)
        l = LOSS(Y, YHAT, MASK) * MASK.sum()
        test_sum += l.item()
        test_cnt += MASK.sum().item()
    print(f"Inference time: {time.time() - start_inf:.2f}s")

test_loss = test_sum / test_cnt
print(f"Best Val={best_val:.4f}, Test={test_loss:.4f}, AvgEpoch={np.mean(times):.2f}s")
