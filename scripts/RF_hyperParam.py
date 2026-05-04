import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

# ── Chargement & préparation ──────────────────────────────────────
df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

df = df_x_final.join(df_y_final, how='inner')
X = df.drop(columns=['quality']).astype(float)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Modèle de base (référence) ────────────────────────────────────
print("=== Modèle de base ===")
rf_base = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
r2_base   = r2_score(y_test, y_pred_base)
mae_base  = mean_absolute_error(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
print(f"R²   : {r2_base:.4f}")
print(f"MAE  : {mae_base:.4f}")
print(f"RMSE : {rmse_base:.4f}")

# ── Grille d'hyperparamètres à explorer ──────────────────────────
param_grid = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', 0.5, None],
    'bootstrap':         [True, False],
}

# ── RandomizedSearchCV — cherche les meilleurs params ────────────
# n_iter=30 → teste 30 combinaisons aléatoires (bon compromis vitesse/qualité)
print("\n=== Recherche des hyperparamètres (patience, ça prend quelques minutes...) ===")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=30,
    cv=5,
    scoring='r2',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print(f"\nMeilleurs paramètres trouvés :")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Meilleur R² en CV : {search.best_score_:.4f}")

# ── Évaluation du meilleur modèle sur le test set ────────────────
rf_best = search.best_estimator_
y_pred_best = rf_best.predict(X_test)
r2_best   = r2_score(y_test, y_pred_best)
mae_best  = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

print(f"\n=== Meilleur modèle — test set ===")
print(f"R²   : {r2_best:.4f}")
print(f"MAE  : {mae_best:.4f}")
print(f"RMSE : {rmse_best:.4f}")

# ── Résumé comparatif ─────────────────────────────────────────────
print("\n══════════════════════════════════════════")
print("           COMPARAISON")
print("══════════════════════════════════════════")
print(f"{'':20} {'Base':>10} {'Optimisé':>10} {'Gain':>10}")
print(f"{'R²':20} {r2_base:>10.4f} {r2_best:>10.4f} {r2_best-r2_base:>+10.4f}")
print(f"{'MAE':20} {mae_base:>10.4f} {mae_best:>10.4f} {mae_best-mae_base:>+10.4f}")
print(f"{'RMSE':20} {rmse_base:>10.4f} {rmse_best:>10.4f} {rmse_best-rmse_base:>+10.4f}")
print("══════════════════════════════════════════")

# ── Figure : base vs optimisé ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, y_pred, label, color in zip(
    axes,
    [y_pred_base, y_pred_best],
    [f'Base — R²={r2_base:.3f} | MAE={mae_base:.2f}',
     f'Optimisé — R²={r2_best:.3f} | MAE={mae_best:.2f}'],
    ['#4C72B0', '#D7263D']
):
    ax.scatter(y_test, y_pred, alpha=0.3, s=12, color=color)
    lim = [-5, 105]
    ax.plot(lim, lim, 'k--', linewidth=1)
    ax.set(title=label, xlabel='Quality réelle', ylabel='Quality prédite')

plt.suptitle('Random Forest — Base vs Hyperparamètres optimisés', fontsize=13)
plt.tight_layout()
plt.savefig("fig_rf_optimise.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] fig_rf_optimise.png")