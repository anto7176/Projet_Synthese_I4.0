import sys, os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Import'))
from import_data_matrice import data_X_formatee, data_Y_formatee

# ---- AFFICHAGE ----
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(data_Y_formatee, lags=24, ax=axes[0], color='#4C72B0')
axes[0].set_title("Autocorrélation de quality (24 heures)")
axes[0].set_xlabel("heures")
axes[0].set_ylabel("Corrélation")

data_Y_formatee.plot(ax=axes[1], color='#4C72B0', linewidth=0.8)
axes[1].set_title("Quality au fil du temps")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Quality")

plt.tight_layout()
plt.savefig("fig_autocorrelation_quality.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_autocorrelation_quality.png")

print("\nCorrélation quality(t) vs quality(t-n) :")
for lag in [1, 2, 3, 6, 12, 24]:
    r = data_Y_formatee.corr(data_Y_formatee.shift(lag))
    print(f"  h - {lag:2d}h : {r:.3f}")
