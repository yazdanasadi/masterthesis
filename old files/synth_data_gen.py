#!/usr/bin/env python
# coding: utf-8
"""
Generates a synthetic ICU-like time series dataset with 3 channels:
- heart_rate, blood_pressure, respiratory_rate
Variable sequence lengths, noise, drop-outs, and mixed archetypes add diversity.
Each patient file is saved as a Parquet in experiments/syntheticdata.
"""

import os
import numpy as np
import pandas as pd

def make_patient_data(patient_id, T_min=80, T_max=200, seed=None):
    """
    Create synthetic time series for one patient with variable length and artifacts.
    Args:
        patient_id (int): unique identifier
        T_min (int): minimum number of time steps
        T_max (int): maximum number of time steps
        seed (int): random seed
    Returns:
        pd.DataFrame with columns ['t','heart_rate','blood_pressure','respiratory_rate']
    """
    if seed is not None:
        np.random.seed(seed + patient_id)
    # Random length
    T = np.random.randint(T_min, T_max+1)
    # Non-uniform timestamps: jitter around uniform grid
    t_lin = np.linspace(0, 1, T)
    jitter = np.random.normal(scale=0.005, size=T)
    t = np.clip(t_lin + jitter, 0, 1)
    t.sort()
    # Choose an archetype: baseline frequency/phases differ
    archetype = np.random.choice(['A','B','C'])
    if archetype == 'A':
        f_hr, f_bp, f_rr = 1.0, 0.7, 1.3
        phase_bp, phase_rr = 0.2, 0.5
        noise_scale = [5, 8, 2]
    elif archetype == 'B':
        f_hr, f_bp, f_rr = 1.5, 1.1, 0.9
        phase_bp, phase_rr = 0.1, 0.3
        noise_scale = [7, 10, 3]
    else:
        f_hr, f_bp, f_rr = 0.8, 1.3, 1.1
        phase_bp, phase_rr = 0.4, 0.7
        noise_scale = [4, 6, 1.5]
    # Simulate channels with baseline + noise
    heart_rate     = 60  + 15 * np.sin(2*np.pi*f_hr*t)         + noise_scale[0]*np.random.randn(T)
    blood_pressure = 120 + 20 * np.sin(2*np.pi*f_bp*t + phase_bp) + noise_scale[1]*np.random.randn(T)
    resp_rate      = 18  +  5 * np.sin(2*np.pi*f_rr*t + phase_rr) + noise_scale[2]*np.random.randn(T)
    df = pd.DataFrame({
        't': t,
        'heart_rate': heart_rate,
        'blood_pressure': blood_pressure,
        'respiratory_rate': resp_rate
    })
    # Introduce random missingness: drop out ~10% points per channel
    for col in ['heart_rate','blood_pressure','respiratory_rate']:
        mask = np.random.rand(T) < 0.1
        df.loc[mask, col] = np.nan
    return df

def main(out_dir='experiments/syntheticdata', n_patients=5000, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    for pid in range(n_patients):
        df = make_patient_data(pid, seed=seed)
        path = os.path.join(out_dir, f'patient_{pid:04d}.parquet')
        df.to_parquet(path)
    print(f"Generated {n_patients} patient files in {out_dir}")

if __name__ == '__main__':
    main()
