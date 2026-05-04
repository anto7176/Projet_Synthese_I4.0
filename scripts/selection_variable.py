import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

# ════════════════════════════════════════════════════════════════════
# MODÈLE
# ════════════════════════════════════════════════════════════════════

df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

df_y_decale = df_y_final.copy()
df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)
df = df_x_final.join(df_y_decale, how='inner')
X = df.drop(columns=['quality']).astype(float)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 1 : RF complet pour obtenir l'ordre d'importance
rf_full = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_full.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'variable':   X.columns,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

variables_triees = importance_df['variable'].tolist()
print("Ordre des variables par importance :")
for i, v in enumerate(variables_triees):
    print(f"  {i+1:2d}. {v}")

# Étape 2 : RF pour chaque sous-ensemble cumulatif
resultats = []

for n in range(1, len(variables_triees) + 1):
    cols = variables_triees[:n]
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train[cols], y_train)
    y_pred = rf.predict(X_test[cols])

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    resultats.append({'n_vars': n, 'derniere_var': cols[-1], 'R2': r2, 'MAE': mae, 'RMSE': rmse})
    print(f"  {n:2d} variables | R²={r2:.4f} | MAE={mae:.4f} | dernière ajoutée : {cols[-1]}")

res_df = pd.DataFrame(resultats)

seuil_r2 = res_df['R2'].max() * 0.99
n_seuil = res_df[res_df['R2'] >= seuil_r2]['n_vars'].min()

print(f"\n→ 99% du R² max atteint avec seulement {n_seuil} variables sur {len(variables_triees)}")

# ════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(res_df['n_vars'], res_df['R2'], color='#4C72B0', marker='o', markersize=5, linewidth=2)
axes[0].axhline(res_df['R2'].max(), color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
axes[0].set_ylabel("R²", fontsize=11)
axes[0].set_title("Performance du Random Forest selon le nombre de variables sélectionnées", fontsize=13)

axes[1].plot(res_df['n_vars'], res_df['MAE'], color='#D7263D', marker='o', markersize=5, linewidth=2)
axes[1].set_ylabel("MAE", fontsize=11)

axes[2].plot(res_df['n_vars'], res_df['RMSE'], color='#2CA02C', marker='o', markersize=5, linewidth=2)
axes[2].set_ylabel("RMSE", fontsize=11)
axes[2].set_xlabel("Nombre de variables", fontsize=11)

for ax in axes:
    ax.set_xticks(res_df['n_vars'])
    ax.set_xticklabels(
        [f"{row['n_vars']}\n{row['derniere_var']}" for _, row in res_df.iterrows()],
        rotation=45, ha='right', fontsize=7
    )
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(n_seuil, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)

axes[0].annotate(f'99% du R² max\natteint avec {n_seuil} variables',
                 xy=(n_seuil, res_df.loc[res_df['n_vars']==n_seuil, 'R2'].values[0]),
                 xytext=(n_seuil + 1, res_df['R2'].min() + 0.01),
                 fontsize=8, color='orange',
                 arrowprops=dict(arrowstyle='->', color='orange'))

plt.tight_layout()
plt.savefig("fig_rf_selection_variables.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"\n[Saved] fig_rf_selection_variables.png")
