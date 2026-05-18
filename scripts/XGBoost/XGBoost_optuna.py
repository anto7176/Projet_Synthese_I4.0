import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Import'))
from import_data_matrice import data_X_formatee, data_Y_formatee

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
N_TRIALS = 150   # ← augmenter pour un search plus exhaustif (ex : 300)

# ════════════════════════════════════════════════════════
# CONSTRUCTION DU DATASET AVEC N LAGS
# ════════════════════════════════════════════════════════

def build_dataset(n_lags: int):
    df = data_X_formatee.copy()
    for lag in range(1, n_lags + 1):
        df[f'quality_lag{lag}'] = data_Y_formatee.shift(lag)
    df = df.dropna()
    y = data_Y_formatee.loc[df.index]
    return df, y

def temporal_split(df, y):
    n = len(df)
    i_val  = int(n * 0.72)
    i_test = int(n * 0.80)
    return (
        df.iloc[:i_val],       y.iloc[:i_val],
        df.iloc[i_val:i_test], y.iloc[i_val:i_test],
        df.iloc[i_test:],      y.iloc[i_test:],
    )

# ════════════════════════════════════════════════════════
# FONCTION OBJECTIF OPTUNA
# ════════════════════════════════════════════════════════

def objective(trial):
    n_lags = trial.suggest_int('n_lags', 1, 6)
    df, y  = build_dataset(n_lags)
    X_tr, y_tr, X_val, y_val, _, _ = temporal_split(df, y)

    params = dict(
        n_estimators          = 3000,
        learning_rate         = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        max_depth             = trial.suggest_int('max_depth', 3, 10),
        subsample             = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree      = trial.suggest_float('colsample_bytree', 0.2, 1.0),
        min_child_weight      = trial.suggest_int('min_child_weight', 1, 10),
        gamma                 = trial.suggest_float('gamma', 0.0, 5.0),
        reg_alpha             = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        reg_lambda            = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        early_stopping_rounds = 50,
        tree_method           = 'hist',
        device                = 'cuda',
        verbosity             = 0,
        random_state          = 42,
    )

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # On sauvegarde le n_estimators optimal (trouvé par early stopping)
    trial.set_user_attr('best_iteration', model.best_iteration)

    y_pred_val = model.predict(X_val)
    return float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

# ════════════════════════════════════════════════════════
# LANCEMENT DE L'OPTIMISATION
# ════════════════════════════════════════════════════════

print(f"Démarrage Optuna — {N_TRIALS} essais (GPU activé)...")

sampler = optuna.samplers.TPESampler(seed=42)
study   = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params   = study.best_params.copy()
n_lags_best   = best_params.pop('n_lags')
n_est_best    = study.best_trial.user_attrs['best_iteration']
rmse_val_best = study.best_value

print(f"\n{'═'*55}")
print(f"  MEILLEURS PARAMÈTRES  (RMSE val = {rmse_val_best:.4f})")
print(f"{'═'*55}")
print(f"  n_lags          : {n_lags_best}")
print(f"  n_estimators    : {n_est_best}  (via early stopping)")
for k, v in best_params.items():
    print(f"  {k:<22}: {v:.6g}" if isinstance(v, float) else f"  {k:<22}: {v}")
print(f"{'═'*55}\n")

# ════════════════════════════════════════════════════════
# MODÈLE FINAL  (entraîné sur train + val, évalué sur test)
# ════════════════════════════════════════════════════════

df_f, y_f = build_dataset(n_lags_best)
X_tr, y_tr, X_val, y_val, X_test, y_test = temporal_split(df_f, y_f)

X_trainval = pd.concat([X_tr, X_val])
y_trainval = pd.concat([y_tr, y_val])

final_model = XGBRegressor(
    **best_params,
    n_estimators = n_est_best,
    tree_method  = 'hist',
    device       = 'cuda',
    verbosity    = 0,
    random_state = 42,
)

print("Entraînement du modèle final (train + val)...")
final_model.fit(X_trainval, y_trainval, verbose=False)

y_pred = final_model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)

print(f"\n── Résultats sur jeu de test ──")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# ════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    f"XGBoost + Optuna — R²={r2:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f}  "
    f"(n_lags={n_lags_best}, n_est={n_est_best})",
    fontsize=12, fontweight='bold'
)

# (1) Prédit vs réel
axes[0].scatter(y_test, y_pred, alpha=0.3, s=12, color='#4C72B0')
axes[0].plot([0, 100], [0, 100], 'r--', linewidth=1)
axes[0].set_xlabel("Quality réelle")
axes[0].set_ylabel("Quality prédite")
axes[0].set_title("Prédit vs Réel (test)")

# (2) Convergence de l'optimisation
trial_values = [t.value for t in study.trials if t.value is not None]
best_so_far  = pd.Series(trial_values).cummin()
axes[1].plot(trial_values, color='#DD8452', linewidth=0.8, alpha=0.6, label='RMSE trial')
axes[1].plot(best_so_far,  color='red',     linewidth=1.5, label=f'Best = {rmse_val_best:.4f}')
axes[1].set_xlabel("Trial n°")
axes[1].set_ylabel("RMSE (validation)")
axes[1].set_title("Convergence Optuna")
axes[1].legend(fontsize=8)

# (3) Top 15 features importantes
feat_imp = pd.Series(final_model.feature_importances_, index=X_trainval.columns)
top15    = feat_imp.nlargest(15)
top15[::-1].plot(kind='barh', ax=axes[2], color='#4C72B0')
axes[2].set_title("Top 15 features importantes")
axes[2].set_xlabel("Importance")

plt.tight_layout()
out = os.path.join(BASE_DIR, "figures", "XGBoost_optuna.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"[Saved] {out}")
