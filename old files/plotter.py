import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hp_results.csv")
pivot = df.pivot_table(index='obs_frac', columns='head_dim', values='best_val_nmse')
pivot.plot(marker='o', title='NMSE vs Obs_frac per Head Dim')
plt.ylabel("Best Validation NMSE")
plt.xlabel("Observation Fraction")
plt.grid(True) 
plt.show()

best_cfg = df.sort_values("best_val_nmse").iloc[0]
print("Best config:", best_cfg.to_dict())
