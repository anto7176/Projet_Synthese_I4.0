import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Import'))
from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

df_x, df_y = charger_donnees(
    os.path.join(BASE_DIR, "data/data_X.csv"),
    os.path.join(BASE_DIR, "data/data_Y.csv"),
)
df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

df_y_decale = df_y_final.copy()
df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)
df = df_x_final.join(df_y_decale, how='inner')

X = df.drop(columns=['quality']).astype(float)
y = df['quality']

print(f"\nDataset : {len(df)} lignes | {X.shape[1]} variables\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# True = données normalisées requises (modèles linéaires)
models = {
    "Régression\nLinéaire":    (LinearRegression(), True),
    "Ridge":                   (Ridge(alpha=1.0), True),
    "Arbre de\nDécision":      (DecisionTreeRegressor(max_depth=12, random_state=42), False),
    "Random\nForest":          (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), False),
    "Gradient\nBoosting":      (GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                           max_depth=5, random_state=42), False),
    "XGBoost":                 (XGBRegressor(n_estimators=1000, learning_rate=0.01,
                                              max_depth=7, subsample=0.7,
                                              colsample_bytree=0.7, gamma=0.2,
                                              reg_alpha=0.01, reg_lambda=5,
                                              random_state=42, verbosity=0), False),
}

results = {}
print(f"{'Modèle':<25} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
print("-" * 53)

for name, (model, use_scaled) in models.items():
    label = name.replace('\n', ' ')
    X_tr = X_train_s if use_scaled else X_train.values
    X_te = X_test_s  if use_scaled else X_test.values

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {"R²": r2, "MAE": mae, "RMSE": rmse}
    print(f"{label:<25} {r2:>8.4f} {mae:>8.2f} {rmse:>8.2f}")

# ---- AFFICHAGE ----
data    = sorted(results.items(), key=lambda x: x[1]["R²"], reverse=True)
names_s = [n         for n, _ in data]
r2_s    = [v["R²"]   for _, v in data]
mae_s   = [v["MAE"]  for _, v in data]
rmse_s  = [v["RMSE"] for _, v in data]

PALETTE = ["#2ecc71", "#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"]
colors  = PALETTE[:len(names_s)]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    "Comparaison des modèles — dataset de base (split 80/20, random_state=42)",
    fontsize=13, fontweight='bold', y=1.01
)

ax1 = axes[0]
bars = ax1.bar(names_s, r2_s, color=colors, edgecolor='white', linewidth=0.6, zorder=3)
ax1.set_ylim(max(0, min(r2_s) - 0.05), 1.02)
ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, zorder=2)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax1.set_ylabel("R²", fontsize=11)
ax1.set_title("R² (plus haut = meilleur)", fontsize=11)
ax1.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
ax1.spines[['top', 'right']].set_visible(False)

for bar, val in zip(bars, r2_s):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.4f}",
        ha='center', va='bottom', fontsize=9, fontweight='bold'
    )

ax2 = axes[1]
x   = np.arange(len(names_s))
w   = 0.35

b1 = ax2.bar(x - w/2, mae_s,  w, label='MAE',  color='#3498db', edgecolor='white', linewidth=0.6, zorder=3)
b2 = ax2.bar(x + w/2, rmse_s, w, label='RMSE', color='#e67e22', edgecolor='white', linewidth=0.6, zorder=3)

ax2.set_xticks(x)
ax2.set_xticklabels(names_s, fontsize=9)
ax2.set_ylabel("Erreur (unités quality 0–100)", fontsize=11)
ax2.set_title("MAE & RMSE (plus bas = meilleur)", fontsize=11)
ax2.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(fontsize=10)

for bar in list(b1) + list(b2):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{bar.get_height():.2f}",
        ha='center', va='bottom', fontsize=8
    )

plt.tight_layout()
out_path = os.path.join(BASE_DIR, "fig_comparaison_modeles.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n[Saved] {out_path}")
