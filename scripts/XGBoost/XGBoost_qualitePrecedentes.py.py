import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Import'))
from import_data_matrice import data_X_formatee, data_Y_formatee

# ════════════════════════════════════════════════════════
# AJOUT DES H-1 et H-2 et H-3 DE QUALITY
# ════════════════════════════════════════════════════════
df = data_X_formatee.copy()
for lag in [1, 2, 3]:
    df[f'quality_lag{lag}'] = data_Y_formatee.shift(lag)

df = df.dropna()
y = data_Y_formatee.loc[df.index]

# ════════════════════════════════════════════════════════
# SPLIT TEMPOREL
# ════════════════════════════════════════════════════════
split      = int(len(df) * 0.8)
val_split  = int(len(df) * 0.72)

X_train = df.iloc[:val_split]
X_val   = df.iloc[val_split:split]
X_test  = df.iloc[split:]
y_train = y.iloc[:val_split]
y_val   = y.iloc[val_split:split]
y_test  = y.iloc[split:]

# ════════════════════════════════════════════════════════
# MODÈLE
# ════════════════════════════════════════════════════════
model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.3,
    min_child_weight=3,
    early_stopping_rounds=50,
    random_state=42,
    verbosity=0,
    tree_method='hist',
    device='cuda'
)

print("Entraînement en cours...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
print(f"Meilleur n_estimators : {model.best_iteration}")

y_pred = model.predict(X_test)

# ════════════════════════════════════════════════════════
# RÉSULTATS
# ════════════════════════════════════════════════════════
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nR²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# ════════════════════════════════════════════════════════
# IMPORTANCE DES LAGS vs reste
# ════════════════════════════════════════════════════════
feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
lag_cols = [f'quality_lag{i}' for i in [1,2,3]]
print("\nImportance des lags quality :")
print(feat_imp[lag_cols].round(4).to_string())
print(f"\nImportance totale lags   : {feat_imp[lag_cols].sum():.4f}")
print(f"Importance reste features: {feat_imp.drop(lag_cols).sum():.4f}")

# ════════════════════════════════════════════════════════
# AFFICHAGE
# ════════════════════════════════════════════════════════
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.3, s=12, color='#4C72B0')
plt.plot([0, 100], [0, 100], 'r--', linewidth=1)
plt.xlabel("Quality réelle")
plt.ylabel("Quality prédite")
plt.title(f"XGBoost + lags — R²={r2:.3f} | MAE={mae:.2f} | RMSE={rmse:.2f}")
plt.tight_layout()
plt.savefig("fig_XGBoost_lags.png", dpi=150, bbox_inches='tight')
plt.show()