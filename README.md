# Inter-Channel Attention for Functional Latent Dynamics in Irregularly Sampled Time Series

This repository is an extended implementation of [Functional-Latent_Dynamics](https://github.com/kloetergensc/Functional-Latent_Dynamics), developed as part of Yazdan Asadi’s Master's Thesis at University of Hildesheim.

It introduces novel model variants like Cross-Channel Attention FLD and integrates several baselines for benchmarking irregularly sampled multivariate time series forecasting on medical datasets such as MIMIC-III, MIMIC-IV, PhysioNet 2012, and USHCN.

---

## Thesis Goal

To improve Functional Latent Dynamics (FLD) for irregular time series forecasting using **Inter-Channel Attention Mechanisms**, and benchmark against state-of-the-art baselines.

---

##  Models 

| Model                 | Description                                                     |
|----------------------|-----------------------------------------------------------------|
| FLD                | Functional Latent Dynamics (original implementation)            |
| CrossAttentionICFLD| FLD with Cross-Channel Attention (proposed in thesis)           |
| MTAN               | Multi-task Attention Networks                                   |
| GraFIT             | Graph-structured time series attention model                    |
| CycleNet           | Flow-based generative time series model                         |

---

##  Datasets

All models are benchmarked on the following datasets using the tsdm library:

- MIMIC-III
- MIMIC-IV
- USHCN (climate)
- PhysioNet 2012

---

## Training Examples

bash


# example for physionet for observing 36 hrs and predicting 12 hrs. All the datasets and models can be run in the similar manner.

python train_FLD.py --epochs 200 --learn-rate 0.001 --batch-size 64 --depth 2 --latent-dim 128 ----num-heads 4 --dataset p12 --fold 0 -ot 36 -ft 12


---

##  Requirements

Install Python dependencies via:

bash
pip install -r requirements.txt


Make sure tsdm is installed (from pip or GitHub):
bash
pip install tsdm
# or
pip install git+https://github.com/leo-prt/tsdm.git


---