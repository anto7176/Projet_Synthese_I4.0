import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

# ── Chargement & préparation ──────────────────────────────────────
# ── Chargement & préparation ──────────────────────────────────────
df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

# Alignement : on garde uniquement les timestamps présents dans Y
# Décalage de Y d'1 heure en arrière — la qualité mesurée à H correspond aux capteurs de H-1
df_y_decale = df_y_final.copy()
df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)
df = df_x_final.join(df_y_decale, how='inner')

print(f"Lignes après alignement X/Y : {len(df)}")

X = df.drop(columns=['quality'])
y = df['quality']

# ── Train / Test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")

# ── Entraînement ──────────────────────────────────────────────────
rf = RandomForestRegressor(
    n_estimators=200,   # nombre d'arbres
    max_depth=None,     # arbres complets
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1           # utilise tous les cœurs CPU
)

rf.fit(X_train, y_train)
print("Modèle entraîné ✓")

# ── Évaluation ────────────────────────────────────────────────────
y_pred = rf.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n── Résultats sur le test set ──")
print(f"MAE  (erreur moyenne)      : {mae:.3f}  (sur une échelle 0-100)")
print(f"RMSE (erreur quadratique)  : {rmse:.3f}")
print(f"R²   (variance expliquée)  : {r2:.3f}  (1.0 = parfait)")

# ── Figure 1 : Réel vs Prédit ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, y_pred, alpha=0.3, s=12, color='#4C72B0')
lim = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
axes[0].plot(lim, lim, 'r--', linewidth=1, label='Prédiction parfaite')
axes[0].set(title='Réel vs Prédit', xlabel='Quality réelle', ylabel='Quality prédite')
axes[0].legend(fontsize=9)

residus = y_test.values - y_pred
axes[1].hist(residus, bins=40, color='#4C72B0', edgecolor='white')
axes[1].axvline(0, color='red', linewidth=1.2, linestyle='--')
axes[1].set(title='Distribution des résidus', xlabel='Résidu (réel - prédit)', ylabel='Fréquence')

plt.suptitle(f'Random Forest — R²={r2:.3f} | MAE={mae:.2f} | RMSE={rmse:.2f}', fontsize=12)
plt.tight_layout()
plt.savefig("fig_rf_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_rf_evaluation.png")

# ── Figure 2 : Importance des variables (Pareto) ──────────────────
importance_df = pd.DataFrame({
    'variable':   X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

importance_df['cumul_%'] = (
    importance_df['importance'].cumsum() / importance_df['importance'].sum() * 100
).round(1)

print(f"\n── Importance des variables ──")
print(importance_df.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(12, 5))

x = np.arange(len(importance_df))
ax1.bar(x, importance_df['importance'], color='#4C72B0', alpha=0.85)
ax1.set_ylabel("Importance (MDI)", color='#4C72B0')
ax1.set_xticks(x)
ax1.set_xticklabels(importance_df['variable'], rotation=45, ha='right', fontsize=9)

ax2 = ax1.twinx()
ax2.plot(x, importance_df['cumul_%'], color='#D7263D', marker='o', markersize=4, linewidth=1.5)
ax2.axhline(80, color='#D7263D', linewidth=1, linestyle='--', alpha=0.6, label='Seuil 80%')
ax2.set_ylabel("Importance cumulée (%)", color='#D7263D')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9)

seuil_idx = (importance_df['cumul_%'] <= 80).sum()
print(f"\n→ {seuil_idx} variables suffisent pour atteindre 80% de l'importance")
print(f"→ Variables dispensables : {importance_df['variable'][seuil_idx:].tolist()}")

ax1.set_title("Diagramme de Pareto — Importance des variables (Random Forest)", fontsize=12)
plt.tight_layout()
plt.savefig("fig_rf_pareto.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_rf_pareto.png")

print("\n── 5 exemples : réel vs prédit ──")
exemples = pd.DataFrame({
    'Quality réelle':  y_test.values[:5],
    'Quality prédite': y_pred[:5].round(1),
    'Écart':           (y_test.values[:5] - y_pred[:5]).round(1)
})
print(exemples.to_string(index=False))