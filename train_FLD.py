#!/usr/bin/env python
# coding: utf-8

"""
Training script for USHCN dataset using FLD model. Supports argument parsing,
configuration loading, dataset preparation, training loop, and validation.
"""

import argparse
import sys
import os
import random
import time
import warnings
import logging
from datetime import datetime
from pathlib import Path
from random import SystemRandom

import numpy as np
import torch
from torch import Tensor, jit
from torch.utils.data import DataLoader
from torch.optim import AdamW
import yaml

# CLI Argument Parsing
parser = argparse.ArgumentParser(description="Training Script for MIMIC dataset.")
parser.add_argument("-q", "--quiet", default=False, const=True, nargs="?")
parser.add_argument("-r", "--run_id", default=None, type=str)
parser.add_argument("-c", "--config", default=None, type=str, nargs=2)
parser.add_argument("-e", "--epochs", default=300, type=int)
parser.add_argument("-f", "--fold", default=0, type=int)
parser.add_argument("-es", "--early-stop", default=30, type=int)
parser.add_argument("-bs", "--batch-size", default=64, type=int)
parser.add_argument("-lr", "--learn-rate", default=0.0001, type=float)
parser.add_argument("-b", "--betas", default=(0.9, 0.999), type=float, nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001, type=float)
parser.add_argument("-ls", "--latent-dim", default=20, type=int)
parser.add_argument("-hd", "--hidden-dim", default=None, type=int)
parser.add_argument("-n", "--note", default="", type=str)
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-dset", "--dataset", default="ushcn", type=str)
parser.add_argument("-ot", "--observation-time", default=24, type=int)
parser.add_argument("-fh", "--forecast-horizon", default=24, type=int)
parser.add_argument("-ed", "--embedding-dim", default=4, type=int)
parser.add_argument("-nh", "--num-heads", default=2, type=int)
parser.add_argument("-dp", "--depth", default=1, type=int)
parser.add_argument("-fn", "--function", default="C", choices=("L", "S", "C", "Q"))

sys.path.insert(0, "../")

try:
    from IPython import get_ipython
    shell = get_ipython()
    ARGS = parser.parse_args("") if shell else parser.parse_args()
except (ImportError, NameError):
    ARGS = parser.parse_args()

# Config from YAML
if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

# Env Setup
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)

# Random Seed
if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

experiment_id = int(SystemRandom().random() * 1e7)
model_path = f"saved_models/FLD-{ARGS.function}_{ARGS.dataset}_{str(experiment_id)}.h5"

# Task Loading
from tsdm.tasks.ushcn.ushcn_debrouwer2019 import USHCN_DeBrouwer2019, ushcn_collate

TASK = USHCN_DeBrouwer2019()
INPUT_DIM = 5

# Dataloader Configs
dloader_config = lambda shuffle: {
    "batch_size": ARGS.batch_size if shuffle else 64,
    "shuffle": shuffle,
    "drop_last": shuffle,
    "pin_memory": True,
    "num_workers": 0 if not shuffle else 4,
    "collate_fn": ushcn_collate,
}

TRAIN_LOADER = TASK.make_dataloader((ARGS.fold, "train"), **dloader_config(True))
VALID_LOADER = TASK.make_dataloader((ARGS.fold, "valid"), **dloader_config(False))
TEST_LOADER = TASK.make_dataloader((ARGS.fold, "test"), **dloader_config(False))

# Loss and Metrics
def MSE(y, yhat, mask): return torch.sum(mask * (y - yhat) ** 2) / torch.sum(mask)
LOSS = jit.script(MSE)

# Model
from src.models.FLD import FLD
MODEL = FLD(INPUT_DIM, ARGS.latent_dim, ARGS.embedding_dim, ARGS.num_heads, ARGS.function, ARGS.depth, DEVICE).to(DEVICE)

# Optimizer
OPTIMIZER = AdamW(MODEL.parameters(), lr=ARGS.learn_rate, weight_decay=ARGS.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, "min", patience=10, factor=0.5, min_lr=5e-4)

# Train/Val/Test Loop
def predict_fn(model, batch):
    T, X, M, TY, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    return Y, model(T, X, M, TY), MY

EARLY_STOP, times, best_val, early_stop = ARGS.early_stop, [], float("inf"), 0
for epoch in range(1, ARGS.epochs + 1):
    MODEL.train()
    train_loss = torch.tensor([LOSS(*predict_fn(MODEL, batch)) for batch in TRAIN_LOADER]).mean()

    MODEL.eval()
    with torch.no_grad():
        val_loss = sum(LOSS(*predict_fn(MODEL, batch)) * batch[-1].sum() for batch in VALID_LOADER)
        val_loss /= sum(batch[-1].sum() for batch in VALID_LOADER)

    print(f"Epoch {epoch}: Train={train_loss:.4f} Val={val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        early_stop = 0
        torch.save({"state_dict": MODEL.state_dict()}, model_path)
    else:
        early_stop += 1
    if early_stop >= EARLY_STOP:
        print(f"Early stopping at epoch {epoch}")
        break
    scheduler.step(val_loss)

# Test
MODEL.load_state_dict(torch.load(model_path)["state_dict"])
MODEL.eval()
with torch.no_grad():
    test_loss = sum(LOSS(*predict_fn(MODEL, batch)) * batch[-1].sum() for batch in TEST_LOADER)
    test_loss /= sum(batch[-1].sum() for batch in TEST_LOADER)
print(f"Final Test Loss: {test_loss:.4f}")
