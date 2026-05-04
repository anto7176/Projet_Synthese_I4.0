import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

# ── Chargement & préparation ──────────────────────────────────────
df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

# ── Décalage de Y d'1 heure en arrière ───────────────────────────
df_y_decale = df_y_final.copy()
df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)

df = df_x_final.join(df_y_decale, how='inner')
print(f"Lignes après alignement : {len(df)}")

# ── 11 variables sélectionnées ────────────────────────────────────
variables_selectionnees = [
    'T_data_3_3', 'T_data_3_1', 'T_data_3_2',
    'H_data', 'T_data_5_2', 'T_data_5_1', 'T_data_5_3',
    'T_data_1_3', 'T_data_1_2', 'T_data_1_1', 'T_data_2_2'
]

X = df[variables_selectionnees].astype(float)
y = df['quality']

# ── Train / Test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
print(f"Features : {X.shape[1]} variables\n")

# ── Modèle de base (référence) ────────────────────────────────────
print("=== XGBoost de base ===")
xgb_base = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
xgb_base.fit(X_train, y_train)
y_pred_base = xgb_base.predict(X_test)
r2_base   = r2_score(y_test, y_pred_base)
mae_base  = mean_absolute_error(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
print(f"R²   : {r2_base:.4f}")
print(f"MAE  : {mae_base:.4f}")
print(f"RMSE : {rmse_base:.4f}")

# ── Grille d'hyperparamètres ──────────────────────────────────────
param_grid = {
    'n_estimators':      [300, 500, 800, 1000],
    'max_depth':         [3, 4, 5, 6, 7, 8],
    'learning_rate':     [0.01, 0.03, 0.05, 0.1, 0.2],
    'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight':  [1, 3, 5, 7],
    'gamma':             [0, 0.1, 0.2, 0.5],
    'reg_alpha':         [0, 0.01, 0.1, 1],
    'reg_lambda':        [0.5, 1, 2, 5],
}

print("\n=== Recherche des hyperparamètres (quelques minutes...) ===")
xgb = XGBRegressor(random_state=42, verbosity=0)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print(f"\nMeilleurs paramètres :")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Meilleur R² en CV : {search.best_score_:.4f}")

# ── Évaluation du meilleur modèle ─────────────────────────────────
xgb_best = search.best_estimator_
y_pred_best = xgb_best.predict(X_test)
r2_best   = r2_score(y_test, y_pred_best)
mae_best  = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

print(f"\n=== Meilleur modèle — test set ===")
print(f"R²   : {r2_best:.4f}")
print(f"MAE  : {mae_best:.4f}")
print(f"RMSE : {rmse_best:.4f}")

# ── 5 exemples ────────────────────────────────────────────────────
print("\n── 5 exemples : réel vs prédit ──")
exemples = pd.DataFrame({
    'Quality réelle':  y_test.values[:5],
    'Quality prédite': y_pred_best[:5].round(1),
    'Écart':           (y_test.values[:5] - y_pred_best[:5]).round(1)
})
print(exemples.to_string(index=False))

# ── Résumé comparatif ─────────────────────────────────────────────
print("\n══════════════════════════════════════════════")
print("              COMPARAISON")
print("══════════════════════════════════════════════")
print(f"{'':20} {'Base':>10} {'Optimisé':>10} {'Gain':>10}")
print(f"{'R²':20} {r2_base:>10.4f} {r2_best:>10.4f} {r2_best-r2_base:>+10.4f}")
print(f"{'MAE':20} {mae_base:>10.4f} {mae_best:>10.4f} {mae_best-mae_base:>+10.4f}")
print(f"{'RMSE':20} {rmse_base:>10.4f} {rmse_best:>10.4f} {rmse_best-rmse_base:>+10.4f}")
print("══════════════════════════════════════════════")

# ── Figure : base vs optimisé ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, y_pred, label, color in zip(
    axes,
    [y_pred_base, y_pred_best],
    [f'Base — R²={r2_base:.3f} | MAE={mae_base:.2f}',
     f'Optimisé — R²={r2_best:.3f} | MAE={mae_best:.2f}'],
    ['#2CA02C', '#D7263D']
):
    ax.scatter(y_test, y_pred, alpha=0.3, s=12, color=color)
    lim = [-5, 105]
    ax.plot(lim, lim, 'k--', linewidth=1)
    ax.set(title=label, xlabel='Quality réelle', ylabel='Quality prédite')

plt.suptitle('XGBoost — Base vs Hyperparamètres optimisés (11 variables + décalage 1h)', fontsize=12)
plt.tight_layout()
plt.savefig("fig_xgboost_optimise.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_xgboost_optimise.png")

# ── Importance des variables ──────────────────────────────────────
importance_df = pd.DataFrame({
    'variable':   X.columns,
    'importance': xgb_best.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

importance_df['cumul_%'] = (
    importance_df['importance'].cumsum() / importance_df['importance'].sum() * 100
).round(1)

print(f"\n── Importance des variables (modèle optimisé) ──")
print(importance_df.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(importance_df))
ax1.bar(x, importance_df['importance'], color='#D7263D', alpha=0.85)
ax1.set_ylabel("Importance", color='#D7263D')
ax1.set_xticks(x)
ax1.set_xticklabels(importance_df['variable'], rotation=45, ha='right', fontsize=9)

ax2 = ax1.twinx()
ax2.plot(x, importance_df['cumul_%'], color='#4C72B0', marker='o', markersize=5, linewidth=1.5)
ax2.axhline(80, color='#4C72B0', linewidth=1, linestyle='--', alpha=0.6, label='Seuil 80%')
ax2.set_ylabel("Importance cumulée (%)", color='#4C72B0')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9)

ax1.set_title("Pareto — Importance des variables (XGBoost optimisé)", fontsize=12)
plt.tight_layout()
plt.savefig("fig_xgboost_optimise_pareto.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_xgboost_optimise_pareto.png")