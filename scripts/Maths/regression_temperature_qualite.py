import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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

y_pred = a * df['T_data_3_1'].values + b
residuals = df['quality'].values - y_pred
sigma = residuals.std()

def plot_with_band(ax, x, y_scatter, x_line, a, b, sigma, title):
    ax.scatter(x, y_scatter, alpha=0.1, s=3, label='Données')
    ax.plot(x_line, a * x_line + b, color='red', linewidth=2,
            label=f'y = {a:.4f}x + {b:.2f}  (R²={r**2:.3f})')
    ax.fill_between(x_line,
                    a * x_line + b - sigma,
                    a * x_line + b + sigma,
                    alpha=0.25, color='red', label=f'±1σ ({sigma:.2f})')
    ax.set_xlabel("T_data_3_1")
    ax.set_ylabel("Qualité (0–100)")
    ax.set_title(title)
    ax.legend(fontsize=8)

# ---- AFFICHAGE ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Avec tous les points
plot_with_band(axes[0], df['T_data_3_1'], df['quality'],
               x_line, a, b, sigma, "Avec tous les points")

# Sans outliers — détection DBSCAN (ε=0.5) sur données normalisées
X2D = StandardScaler().fit_transform(df[['T_data_3_1', 'quality']].values)
labels = DBSCAN(eps=0.05, min_samples=5).fit_predict(X2D)
mask = labels != -1
df_clean = df[mask]
a_c, b_c, r_c, _, _ = stats.linregress(df_clean['T_data_3_1'], df_clean['quality'])
sigma_c = (df_clean['quality'].values - (a_c * df_clean['T_data_3_1'].values + b_c)).std()
x_line_c = np.linspace(df_clean['T_data_3_1'].min(), df_clean['T_data_3_1'].max(), 200)
n_out = (~mask).sum()
print(f"DBSCAN : {n_out} outliers retirés / {len(df)} pts ({100*n_out/len(df):.1f}%)")
print(f"Régression nettoyée : qualité = {a_c:.6f} × T_data_3_1 + {b_c:.4f}  (R²={r_c**2:.4f})")

plot_with_band(axes[1], df_clean['T_data_3_1'], df_clean['quality'],
               x_line_c, a_c, b_c, sigma_c,
               f"Sans outliers DBSCAN  —  {n_out} retirés")

plt.suptitle("Régression linéaire : Qualité vs T_data_3_1", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
