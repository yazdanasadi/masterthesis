# Inter-Channel Attention for Functional Latent Dynamics in Irregularly Sampled Time Series

This repository is an extended implementation of [Functional-Latent_Dynamics](https://github.com/kloetergensc/Functional-Latent_Dynamics), developed as part of Yazdan Asadi’s Master's Thesis at the University of Hildesheim.

It introduces novel model variants based on **Inter-Channel Functional Latent Dynamics (ICFLD)**, integrating multiple attention mechanisms and residual cycle forecasting for benchmarking irregularly sampled multivariate time series forecasting on real-world datasets such as MIMIC-III, MIMIC-IV, PhysioNet 2012, and USHCN.

---

## Thesis Goal

To improve Functional Latent Dynamics (FLD) for irregular time series forecasting by introducing **Inter-Channel Attention Mechanisms**, including **Residual Cycle Forecasting (RCF)**, and benchmarking against competitive baselines.

---

## Models

| Model       | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `ICFLD`     | Original Functional Latent Dynamics with inter-channel attention            |
| `ICFLD-RCF` | ICFLD extended with **Residual Cycle Forecasting** (CycleNet variant)       |
| `FLD`       | Functional Latent Dynamics                                              |
| `MTAN`      | Multi-task Attention Networks                                               |
| `GraFIT`    | Graph-structured attention model for irregular multivariate time series     |

---

### 🌀 ICFLD-RCF: Residual Cycle Forecasting

`ICFLD-RCF` introduces **Residual Cycle Forecasting (RCF)** on top of ICFLD to explicitly remove and reintroduce periodic components in time series data.

- **RCF Mechanism:**  
  Cyclical signals (e.g., circadian rhythms or seasonal cycles) are estimated and **removed from the input** before passing through the attention layers. The model then focuses on the residual signal dynamics. At the end of the pipeline, the cycles are **added back** to reconstruct the final forecast.

- **Usage:**  
  To enable this behavior, set the `--cycle` flag when running the ICFLD model.

```bash
# Example: Train ICFLD with RCF on MIMIC-III
python train_grafiti.py --model icfld --cycle --dataset mimic3 --epochs 300 --batch-size 64
```

If `--cycle` is omitted, ICFLD runs in standard mode (without residual cycle removal).

---

## Datasets

All ICFLD variants and baseline models are benchmarked using the [tsdm](https://github.com/ETHZ-TIK/tsdm) framework on:

- **MIMIC-III**: De-identified ICU patient records  
- **MIMIC-IV**: Expanded ICU dataset with improved coverage  
- **USHCN**: U.S. Historical Climatology Network (climate data)  
- **PhysioNet 2012**: ICU challenge dataset with multivariate physiological signals  

---

## Training Examples

```bash
# Train standard ICFLD on PhysioNet: 36h observation, 12h forecast
python train_fld.py --model icfld --epochs 200 --learn-rate 0.001 --batch-size 64 \
--depth 2 --latent-dim 128 --num-heads 4 --dataset p12 --fold 0 -ot 36 -ft 12
```

```bash
# Train ICFLD-RCF (cycle forecasting enabled) on MIMIC-III
python train_grafiti.py --model icfld --cycle --dataset mimic3 --epochs 300 --batch-size 64
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Install `tsdm`:

```bash
pip install tsdm
```
