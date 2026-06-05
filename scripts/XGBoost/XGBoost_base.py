import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Import'))
from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

VARIABLES = [
    'T_data_3_3',
    'T_data_3_1',
    'T_data_3_2',
    'H_data',
    'T_data_5_2',
    'T_data_5_1',
    'T_data_5_3',
    'T_data_1_3',
    'T_data_1_2',
    'T_data_1_1',
    'T_data_2_2',
]

df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")
df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

df_y_decale = df_y_final.copy()
df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)

df = df_x_final.join(df_y_decale, how='inner')

X = df[VARIABLES].astype(float)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.2,
    reg_alpha=0.01,
    reg_lambda=5,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nVariables utilisées ({len(VARIABLES)}) : {VARIABLES}")
print(f"\nR²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

print("\n5 exemples : réel vs prédit")
exemples = pd.DataFrame({
    'Réel':   y_test.values[:5].round(1),
    'Prédit': y_pred[:5].round(1),
    'Écart':  (y_test.values[:5] - y_pred[:5]).round(1)
})
print(exemples.to_string(index=False))

# ---- AFFICHAGE ----
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

bins   = [0, 20, 40, 60, 80, 100]
labels = ['Très basse\n(0-20)', 'Basse\n(20-40)', 'Moyenne\n(40-60)', 'Haute\n(60-80)', 'Très haute\n(80-100)']

y_reel_classes = pd.cut(y_test.values, bins=bins, labels=labels, include_lowest=True)
y_pred_classes = pd.cut(y_pred,        bins=bins, labels=labels, include_lowest=True)

cm = confusion_matrix(y_reel_classes, y_pred_classes, labels=labels)

# Normalisation par ligne pour obtenir le % par classe réelle
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

annots = np.array([
    [f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(len(labels))]
    for i in range(len(labels))
])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_pct, annot=annots, fmt='', cmap='Blues',
    xticklabels=labels, yticklabels=labels,
    ax=ax, linewidths=0.5, linecolor='white',
    vmin=0, vmax=100
)
ax.set_xlabel("Quality prédite", fontsize=12)
ax.set_ylabel("Quality réelle",  fontsize=12)
ax.set_title(f"Matrice de confusion — 5 classes\nMAE={mae:.2f} | R²={r2:.3f}", fontsize=13)
plt.tight_layout()
plt.savefig("fig_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_confusion_matrix.png")
