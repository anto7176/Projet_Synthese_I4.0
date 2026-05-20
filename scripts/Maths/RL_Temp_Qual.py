import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Import'))
from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

df_x, df_y = charger_donnees(
    os.path.join(BASE_DIR, "data/data_X.csv"),
    os.path.join(BASE_DIR, "data/data_Y.csv"),
)

df_x = nettoyer_donnees(df_x)
df_y = normaliser_qualite(df_y, colonne='quality')
df_x = formater_index_temporel(df_x, "date_time")
df_y = formater_index_temporel(df_y, "date_time")

df_t = df_x[['T_data_3_1']].resample('h').mean()
df_q = df_y[['quality']].copy()
df_q.index = df_q.index.floor('h')

df = df_q.join(df_t, how='inner').dropna()

# Régression linéaire
a, b, r, p, _ = stats.linregress(df['T_data_3_1'], df['quality'])
print(f"Régression : qualité = {a:.6f} × T_data_3_1 + {b:.4f}")
print(f"R² = {r**2:.4f}")

x_line = np.linspace(df['T_data_3_1'].min(), df['T_data_3_1'].max(), 200)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['T_data_3_1'], df['quality'], alpha=0.1, s=3, label='Données')
ax.plot(x_line, a * x_line + b, color='red', linewidth=2,
        label=f'y = {a:.4f}x + {b:.2f}  (R²={r**2:.3f})')
ax.set_xlabel("T_data_3_1")
ax.set_ylabel("Qualité (0–100)")
ax.set_title("Qualité vs T_data_3_1")
ax.legend()
plt.tight_layout()
plt.show()
