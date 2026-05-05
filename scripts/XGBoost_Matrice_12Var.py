import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Importation directe de la DB formatée et sauvegardée
from import_data_matrice import data_X_formatee, data_Y_formatee

# ════════════════════════════════════════════════════════
# SPLIT
# ════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(data_X_formatee, data_Y_formatee, test_size=0.2, random_state=42)

# ════════════════════════════════════════════════════════
# MEILLEURS PARAMÈTRES (trouvés par RandomizedSearchCV)
# ════════════════════════════════════════════════════════
model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.3,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    min_child_weight=3,
    random_state=42,
    verbosity=0,
    tree_method='hist',
    device='cuda'
)

print("Entraînement en cours...")
model.fit(X_train, y_train)
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

print("\n── 5 exemples : réel vs prédit ──")
exemples = pd.DataFrame({
    'Réel':   y_test.values[:5].round(1),
    'Prédit': y_pred[:5].round(1),
    'Écart':  (y_test.values[:5] - y_pred[:5]).round(1)
})
print(exemples.to_string(index=False))