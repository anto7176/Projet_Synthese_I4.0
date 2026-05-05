import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
# PROCESS MATRIX — 12 variables, minutes 5 à 59
# ════════════════════════════════════════════════════════
VARIABLES = [
    'T_data_3_3', 'T_data_3_1', 'T_data_3_2',
    'H_data', 'T_data_5_2', 'T_data_5_1', 'T_data_5_3',
    'T_data_1_3', 'T_data_1_2', 'T_data_1_1', 'T_data_2_2', 'T_data_2_3'
]

print("Construction de la process matrix...")
df_x_pm = df_x_final[VARIABLES].copy()
df_x_pm['_heure']  = df_x_final.index.floor('h')
df_x_pm['_minute'] = df_x_final.index.minute
df_x_pm = df_x_pm[df_x_pm['_minute'] >= 5]

df_pivot = df_x_pm.pivot_table(
    index='_heure', columns='_minute',
    values=VARIABLES, aggfunc='first'
)
df_pivot.columns = [f"{var}_m{int(m):02d}" for var, m in df_pivot.columns]
df_pivot.index.name = 'date_time'

df_y_tronque = df_y_decale.copy()
df_y_tronque.index = (df_y_tronque.index - pd.Timedelta(minutes=5)).floor('h')

df_final = df_pivot.join(df_y_tronque, how='inner').dropna()

X = df_final.drop(columns=['quality']).astype(float)
y = df_final['quality']

print(f"Dataset : {len(df_final)} lignes | {X.shape[1]} variables")

# ════════════════════════════════════════════════════════
# MODÈLE
# ════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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