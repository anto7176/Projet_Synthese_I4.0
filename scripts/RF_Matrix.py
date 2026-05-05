import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ════════════════════════════════════════════════════════
# CHARGEMENT 
# ════════════════════════════════════════════════════════
from import_data_matrice import data_X_formatee, data_Y_formatee

print(f"Dataset : {len(data_X_formatee)} lignes | {data_X_formatee.shape[1]} variables")

# ════════════════════════════════════════════════════════
# MODÈLE
# ════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(data_X_formatee, data_Y_formatee, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=300, min_samples_split= 2,  min_samples_leaf= 1,max_features= 0.5,n_jobs=-1)

print("Entraînement du modèle Random Forest en cours...")
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²   = {r2:.3f}")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")

# ════════════════════════════════════════════════════════
# AFFICHAGE
# ════════════════════════════════════════════════════════
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.3, s=12, color='#4C72B0')
plt.plot([0, 100], [0, 100], 'r--', linewidth=1)
plt.xlabel("Quality réelle")
plt.ylabel("Quality prédite")
plt.title(f"Random Forest — R²={r2:.3f} | MAE={mae:.2f} | RMSE={rmse:.2f}")
plt.tight_layout()
plt.savefig("fig_rf1.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_rf1.png")