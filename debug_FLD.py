#!/usr/bin/env python
# coding: utf-8

# # MIMIC-III

# ## Input Parsing (for command line use)

# In[ ]:


import argparse
import sys

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for MIMIC dataset.")
parser.add_argument("-q",  "--quiet",        default=False,  const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r",  "--run_id",       default=None,   type=str,   help="run_id")
parser.add_argument("-c",  "--config",       default=None,   type=str,   help="load external config", nargs=2)
parser.add_argument("-e",  "--epochs",       default=300,    type=int,   help="maximum epochs")
parser.add_argument("-f",  "--fold",         default=0,      type=int,   help="fold number")
parser.add_argument("-es", "--early-stop", default=30)
parser.add_argument("-bs", "--batch-size",   default=64,     type=int,   help="batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.0001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-ls", "--latent-dim",  default=20,    type=int,   help="dimensionality of latent time seris produced by mtan encoder")
parser.add_argument("-hd", "--hidden-dim", default=None, type=int, help="width of hidden layer in output network")
parser.add_argument("-n",  "--note",         default="",     type=str,   help="Note that can be added")
parser.add_argument("-s",  "--seed",         default=0,   type=int,   help="Set the random seed.")
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-ot","--observation-time",default=24, type=int, help ="obersvation time in hours")
parser.add_argument("-fh","--forecast-horizon",default=24, type=int, help ="forcasting horizon in hours")
parser.add_argument("-ed", "--embedding-dim",default=4, type=int, help="embeddeding dimension for the attention")
parser.add_argument("-nh", "--num-heads", default = 2, type=int)
parser.add_argument("-dp", "--depth", default=1, type=int)
parser.add_argument("-fn", "--function", default="C", choices=("L","S","C","Q"), help="type of hidden function (L: linear, S: sine, C: constant)")
# fmt: on
sys.path.insert(0, "../")

print(" ".join(sys.argv))


try:
    get_ipython().run_line_magic(
        "config", "InteractiveShell.ast_node_interactivity='last_expr_or_assign'"
    )
except NameError:
    ARGS = parser.parse_args()
else:
    ARGS = parser.parse_args("")

import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

import logging
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from random import SystemRandom

import numpy as np
import torch
import torchinfo
from IPython.core.display import HTML
from torch import Tensor, jit
from tqdm.autonotebook import tqdm, trange

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")


if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_STR = (
    f"f={ARGS.fold}_bs={ARGS.batch_size}_lr={ARGS.learn_rate}_ls={ARGS.latent_dim}"
)
RUN_ID = ARGS.run_id or datetime.now().isoformat(timespec="seconds")
CFG_ID = 0 if ARGS.config is None else ARGS.config[1]
HOME = Path.cwd()
print(DEVICE)

experiment_id = 9436521
print(ARGS, experiment_id)

model_path = ("saved_models/" + ARGS.dataset + "_" + str(experiment_id) + ".h5",)

from tsdm.tasks.mimic_iii_debrouwer2019 import mimic_collate as task_collate_fn

if ARGS.dataset == "mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

    TASK = MIMIC_III_DeBrouwer2019(
        condition_time=ARGS.observation_time, forecast_horizon=ARGS.forecast_horizon
    )
    INPUT_DIM = 96


if ARGS.dataset == "mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

    TASK = MIMIC_IV_Bilos2021(
        condition_time=ARGS.observation_time, forecast_horizon=ARGS.forecast_horizon
    )
    INPUT_DIM = 102

if ARGS.dataset == "p12":
    from tsdm.tasks.physionet2012 import Physionet2012

    TASK = Physionet2012(
        condition_time=ARGS.observation_time, forecast_horizon=ARGS.forecast_horizon
    )
    INPUT_DIM = 37

if ARGS.dataset == "ushcn":
    from tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019

    TASK = USHCN_DeBrouwer2019(
        condition_time=ARGS.observation_time, forecast_horizon=ARGS.forecast_horizon
    )
    INPUT_DIM = 5

# ## Initialize DataLoaders

# In[ ]:


dloader_config_train = {
    "batch_size": ARGS.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 4,
    "collate_fn": task_collate_fn,
}

dloader_config_infer = {
    "batch_size": 256,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": task_collate_fn,
}

TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
INFER_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_infer)
VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)
EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}


def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sum(mask * ((y - yhat) ** 2)) / torch.sum(mask)
    return err


def MAE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sum(mask * torch.abs(y - yhat), 1) / (torch.sum(mask, 1))
    return torch.mean(err)


def RMSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sqrt(torch.sum(mask * (y - yhat) ** 2, 1) / (torch.sum(mask, 1)))
    return torch.mean(err)


METRICS = {
    # "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    # "MAE": jit.script(MAE),
}
LOSS = jit.script(MSE)


from src.models.FLD import FLD

MODEL = FLD(
    input_dim=INPUT_DIM,
    latent_dim=ARGS.latent_dim,
    embed_dim_per_head=ARGS.embedding_dim,
    num_heads=ARGS.num_heads,
    function=ARGS.function,
    depth=ARGS.depth,
    device=DEVICE,
).to(DEVICE)


def predict_fn(model, batch) -> tuple[Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, TY, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    YHAT = model(T, X, M, TY)
    # return torch.masked_select(Y,MY), torch.masked_select(YHAT,MY)
    return Y, YHAT, MY


batch = next(iter(TRAIN_LOADER))
MODEL.zero_grad(set_to_none=True)

# Forward
Y, YHAT, MASK = predict_fn(MODEL, batch)
# Backward
R = LOSS(Y, YHAT, MASK)
assert torch.isfinite(R).item(), "Model Collapsed!"
R.backward()

# Reset
MODEL.zero_grad(set_to_none=True)

from torch.optim import AdamW

EARLY_STOP = ARGS.early_stop

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER, "min", patience=10, factor=0.5, min_lr=0.0005, verbose=True
)

times = []
ovr_train_start = time.time()
es = False
best_val_loss = 10e8
total_num_batches = 0
N_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
print("#PARAMS: ", N_params)
for epoch in range(1, ARGS.epochs + 1):
    if True:
        break
    loss_list = []
    start_time = time.time()
    for batch in TRAIN_LOADER:
        total_num_batches += 1
        OPTIMIZER.zero_grad()
        Y, YHAT, MASK = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT, MASK)
        assert torch.isfinite(R).item(), "Model Collapsed!"
        loss_list.append([R])
        # Backward
        R.backward()
        OPTIMIZER.step()
    epoch_time = time.time()
    train_loss = torch.mean(torch.Tensor(loss_list))
    loss_list = []
    count = 0
    with torch.no_grad():
        for batch in VALID_LOADER:
            total_num_batches += 1
            # Forward
            Y, YHAT, MASK = predict_fn(MODEL, batch)
            R = LOSS(Y, YHAT, MASK)
            # if R.isnan():
            #   pdb.set_trace()
            loss_list.append([R * MASK.sum()])
            count += MASK.sum()
    val_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
    times.append(epoch_time - start_time)
    print(
        epoch,
        "Train: ",
        train_loss.item(),
        " VAL: ",
        val_loss.item(),
        " epoch time: ",
        (epoch_time - start_time),
        "secs",
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "args": ARGS,
                "epoch": epoch,
                "state_dict": MODEL.state_dict(),
                "optimizer_state_dict": OPTIMIZER.state_dict(),
                "loss": train_loss,
            },
            model_path,
        )
        early_stop = 0
    else:
        early_stop += 1
    if early_stop == EARLY_STOP:
        print(
            f"Early stopping because of no improvement in val. metric for {EARLY_STOP} epochs"
        )
        es = True
    scheduler.step(val_loss)

    # LOGGER.log_epoch_end(epoch)
# chp = torch.load(model_path)
# MODEL.load_state_dict(chp["state_dict"])
loss_list = []
count = 0
with torch.no_grad():
    inf_start = time.time()
    for batch in TEST_LOADER:
        total_num_batches += 1
        # Forward
        Y, YHAT, MASK = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT, MASK)
        assert torch.isfinite(R).item(), "Model Collapsed!"
        # loss_list.append([R*Y.shape[0]])
        loss_list.append([R * MASK.sum()])
        count += MASK.sum()
    print(f"inference_time: {time.time() - inf_start}")
test_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
print(f"best_val_loss: {best_val_loss.item()},  test_loss: {test_loss.item()}")
print(f"avg_epoch_time: {np.mean(times)}")
