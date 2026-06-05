import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
os.chdir(BASE_DIR)

sys.path.insert(0, os.path.join(BASE_DIR, 'scripts', 'Import'))
from import_data_matrice import data_X_formatee, data_Y_formatee

if data_X_formatee is None:
    raise RuntimeError("Impossible de charger la process matrix. Vérifiez les fichiers data/.")

X = data_X_formatee
y = data_Y_formatee

print(f"\nDataset (process matrix) : {len(X)} lignes | {X.shape[1]} features\n")

# Température moyenne chambre C3 convertie en Celsius (données en Fahrenheit)
cols_c3 = [c for c in X.columns if c.startswith('T_data_3_')]
if not cols_c3:
    raise RuntimeError("Aucune colonne C3 (T_data_3_*) trouvée dans la process matrix.")

temp_c3 = ((X[cols_c3] - 32) / 1.8).mean(axis=1)

# Tranche de température avec la qualité moyenne la plus élevée (effectif >= 30)
MIN_COUNT = 30

df = pd.DataFrame({"temp": temp_c3.values, "qualite": y.values})
df["bin"] = df["temp"].round().astype(int)

agg = (df.groupby("bin")
         .agg(temp_moy=("temp", "mean"),
              qualite_moy=("qualite", "mean"),
              n=("qualite", "size"))
         .query("n >= @MIN_COUNT"))

opt_bin  = agg["qualite_moy"].idxmax()
temp_opt = agg.loc[opt_bin, "temp_moy"]

qualites = df.loc[df["bin"] == opt_bin, "qualite"].values

stat = {
    "Effectif (n)": len(qualites),
    "Moyenne":      qualites.mean(),
    "Médiane":      np.median(qualites),
    "Min":          qualites.min(),
    "Max":          qualites.max(),
    "Écart-type":   qualites.std(),
}

print("=" * 60)
print(f"  QUALITÉ À LA TEMPÉRATURE C3 OPTIMALE ≈ {temp_opt:.1f} °C")
print("=" * 60)
for k, v in stat.items():
    print(f"  {k:<14} {v:>8.2f}")
print("=" * 60)

# ---- AFFICHAGE ----
fig, ax = plt.subplots(figsize=(6.5, 7))

bp = ax.boxplot(
    qualites, widths=0.45, patch_artist=True, showmeans=True,
    boxprops=dict(facecolor='#2ecc71', alpha=0.45, edgecolor='#1e8449'),
    medianprops=dict(color='black', linewidth=2),
    meanprops=dict(marker='D', markerfacecolor='#f1c40f',
                   markeredgecolor='black', markersize=9),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker='o', markersize=4, alpha=0.4, markeredgecolor='gray'),
)

# Jitter horizontal pour visualiser la distribution individuelle
rng = np.random.default_rng(42)
x_jit = rng.normal(1.0, 0.045, size=len(qualites))
ax.scatter(x_jit, qualites, s=22, alpha=0.55, color='#27ae60',
           edgecolor='white', linewidth=0.4, zorder=3)

ax.set_xticks([1])
ax.set_xticklabels([f"T_C3 ≈ {temp_opt:.1f} °C"], fontsize=11)
ax.set_ylabel("Qualité (0–100)", fontsize=11)
ax.set_title(f"Distribution de la qualité à la température C3 optimale\n(≈ {temp_opt:.1f} °C)",
             fontsize=12, fontweight='bold')
ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
ax.spines[['top', 'right']].set_visible(False)

txt = "\n".join([
    f"n          = {stat['Effectif (n)']:.0f}",
    f"Moyenne    = {stat['Moyenne']:.1f}",
    f"Médiane    = {stat['Médiane']:.1f}",
    f"Min        = {stat['Min']:.1f}",
    f"Max        = {stat['Max']:.1f}",
    f"Écart-type = {stat['Écart-type']:.1f}",
])
ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=10, family='monospace',
        va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#bbbbbb', alpha=0.9))

ax.plot([], [], color='black', linewidth=2, label='Médiane')
ax.plot([], [], marker='D', markerfacecolor='#f1c40f', markeredgecolor='black',
        linestyle='None', markersize=9, label='Moyenne')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
out_path = os.path.join(BASE_DIR, "fig_c3_qualite_temp_optimale.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n[Saved] {out_path}")
