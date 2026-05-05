import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

# ════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════
df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")
df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

df_y_decale = df_y_final.copy()
df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)

# ════════════════════════════════════════════════════════
# PROCESS MATRIX
# ════════════════════════════════════════════════════════
print("Construction de la process matrix...")
df_x_pm = df_x_final.copy()
df_x_pm['_heure']  = df_x_pm.index.floor('h')
df_x_pm['_minute'] = df_x_pm.index.minute
df_x_pm = df_x_pm[df_x_pm['_minute'] < 60]

feature_cols = [c for c in df_x_final.columns]
df_pivot = df_x_pm.pivot_table(
    index='_heure', columns='_minute',
    values=feature_cols, aggfunc='first'
)
df_pivot.columns = [f"{var}_m{int(m):02d}" for var, m in df_pivot.columns]
df_pivot.index.name = 'date_time'

df_y_tronque = df_y_decale.copy()
df_y_tronque.index = df_y_tronque.index.floor('h')

df_final = df_pivot.join(df_y_tronque, how='inner').dropna()
print(f"Process matrix : {df_final.shape[0]} lignes × {df_final.shape[1]-1} features")

X = df_final.drop(columns=['quality']).astype(float)
y = df_final['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ════════════════════════════════════════════════════════
# MODÈLE DE BASE (référence)
# ════════════════════════════════════════════════════════
print("\n=== Modèle de base ===")
base = XGBRegressor(
    n_estimators=1000, learning_rate=0.01, max_depth=7,
    subsample=0.7, colsample_bytree=0.3,
    gamma=0.2, reg_alpha=0.01, reg_lambda=5,
    random_state=42, verbosity=0, n_jobs=-1
)
base.fit(X_train, y_train)
y_pred_base = base.predict(X_test)
r2_base  = r2_score(y_test, y_pred_base)
mae_base = mean_absolute_error(y_test, y_pred_base)
print(f"R²  : {r2_base:.4f} | MAE : {mae_base:.4f}")

# ════════════════════════════════════════════════════════
# RECHERCHE HYPERPARAMÈTRES
# ════════════════════════════════════════════════════════
param_grid = {
    'n_estimators':     [500, 1000, 2000],
    'max_depth':        [4, 5, 6, 7, 8],
    'learning_rate':    [0.005, 0.01, 0.03, 0.05],
    'subsample':        [0.6, 0.7, 0.8],
    'colsample_bytree': [0.2, 0.3, 0.4, 0.5],  # faible car ~1000 features
    'gamma':            [0, 0.1, 0.2, 0.5],
    'reg_alpha':        [0, 0.01, 0.1, 1],
    'reg_lambda':       [1, 5, 10],
    'min_child_weight': [1, 3, 5],
}

print("\n=== Recherche hyperparamètres (long, patience...) ===")
search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=30,           # réduit car chaque fit est plus long
    cv=3,                # 3 folds au lieu de 5 pour aller plus vite
    scoring='r2',
    verbose=3,           # Passé à 3 pour afficher la progression détaillée entre chaque test
    random_state=42,
    n_jobs=1             # 1 job car XGBoost utilise déjà tous les cœurs
)
search.fit(X_train, y_train)

print(f"\nMeilleurs paramètres :")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Meilleur R² en CV : {search.best_score_:.4f}")

# ════════════════════════════════════════════════════════
# ÉVALUATION
# ════════════════════════════════════════════════════════
y_pred_best = search.best_estimator_.predict(X_test)
r2_best  = r2_score(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

print(f"\n=== Résultats ===")
print(f"R²   : {r2_best:.4f}")
print(f"MAE  : {mae_best:.4f}")
print(f"RMSE : {rmse_best:.4f}")

print("\n══════════════════════════════════════════════════")
print("                  COMPARAISON")
print("══════════════════════════════════════════════════")
print(f"{'':30} {'R²':>8} {'MAE':>8}")
print(f"{'XGBoost 11 vars':30} {'0.9263':>8} {'3.1831':>8}")
print(f"{'Process matrix base':30} {r2_base:>8.4f} {mae_base:>8.4f}")
print(f"{'Process matrix optimisé':30} {r2_best:>8.4f} {mae_best:>8.4f}")
print("══════════════════════════════════════════════════")