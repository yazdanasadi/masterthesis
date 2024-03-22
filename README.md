# Functional Latent Dynamics

This is the code of ECML-PAKDD 2024 submission: Functional Latent Dynamics for Irregularly Sampled Time Series Forecasting

# Requirements
Please refer to the requirements.txt file

# Training on Goodwin dataset
To rerun the experiment on the Goodwin Dataset, please run the Goodwin.ipynb notebook.

# Training and Evaluation on Benchmark Models

We provide an example for ``physionet`` for observing 36 hrs and predicting 12 hrs. All the datasets and models can be run in the similar manner.

```
train_FLD.py --epochs 200 --learn-rate 0.001 --batch-size 64 --depth 2 --latent-dim 128 ----num-heads 4 --dataset p12 --fold 0 -ot 36 -ft 12
```

Remaining datasets can be run similarly. MIMIC-IV and MIMIC-III require permissions to download the data. Once, the datasets are downloaded, you can add them to the folder .tsdm/rawdata/ and use the TSDM package to extract the folds. 
We use TSDM package provided by Scholz .et .al from [https://openreview.net/forum?id=a-bD9-0ycs0]
