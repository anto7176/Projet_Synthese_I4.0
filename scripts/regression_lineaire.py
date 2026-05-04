import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

# ── Chargement & préparation ──────────────────────────────────────
df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

df = df_x_final.join(df_y_final, how='inner')
print(f"Lignes après alignement : {len(df)}")

X = df.drop(columns=['quality'])
y = df['quality']

# ── Train / Test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Normalisation des features (obligatoire pour la régression linéaire) ──
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Entraînement ──────────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
print("Modèle entraîné ✓")

# ── Évaluation ────────────────────────────────────────────────────
y_pred = lr.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n── Résultats sur le test set ──")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# ── 5 exemples ────────────────────────────────────────────────────
print("\n── 5 exemples : réel vs prédit ──")
exemples = pd.DataFrame({
    'Quality réelle':  y_test.values[:5],
    'Quality prédite': y_pred[:5].round(1),
    'Écart':           (y_test.values[:5] - y_pred[:5]).round(1)
})
print(exemples.to_string(index=False))

# ── Coefficients ──────────────────────────────────────────────────
coef_df = pd.DataFrame({
    'variable':   X.columns,
    'coefficient': lr.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\n── Coefficients (impact sur quality) ──")
print(coef_df.to_string(index=False))

# ── Figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, y_pred, alpha=0.3, s=12, color='#D7263D')
lim = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
axes[0].plot(lim, lim, 'k--', linewidth=1, label='Prédiction parfaite')
axes[0].set(title='Réel vs Prédit', xlabel='Quality réelle', ylabel='Quality prédite')
axes[0].legend(fontsize=9)

residus = y_test.values - y_pred
axes[1].hist(residus, bins=40, color='#D7263D', edgecolor='white')
axes[1].axvline(0, color='black', linewidth=1.2, linestyle='--')
axes[1].set(title='Distribution des résidus', xlabel='Résidu (réel - prédit)', ylabel='Fréquence')

plt.suptitle(f'Régression Linéaire — R²={r2:.3f} | MAE={mae:.2f} | RMSE={rmse:.2f}', fontsize=12)
plt.tight_layout()
plt.savefig("fig_regression_lineaire.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_regression_lineaire.png")