import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

# ════════════════════════════════════════════════════════════════════
# MODÈLE
# ════════════════════════════════════════════════════════════════════

df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

# Décalage de Y d'1 heure en arrière (la qualité à H correspond aux capteurs de H-1)
df_y_decale = df_y_final.copy()
df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)

# Option 1 : join exact
df_exact = df_x_final.join(df_y_decale, how='inner')
X_exact = df_exact.drop(columns=['quality']).astype(float)
y_exact = df_exact['quality']
print(f"Option 1 (join exact)     : {len(df_exact)} lignes")

# Option 2 : agrégation horaire de X
df_x_tronque = df_x_final.copy()
df_x_tronque.index = df_x_tronque.index.floor('h')

df_x_horaire = df_x_tronque.groupby(level=0).agg(['mean', 'max', 'min', 'std'])
df_x_horaire.columns = ['_'.join(col) for col in df_x_horaire.columns]

df_y_tronque = df_y_decale.copy()
df_y_tronque.index = df_y_tronque.index.floor('h')

df_horaire = df_x_horaire.join(df_y_tronque, how='inner').dropna()
X_horaire = df_horaire.drop(columns=['quality']).astype(float)
y_horaire = df_horaire['quality']
print(f"Option 2 (agrégation 1h)  : {len(df_horaire)} lignes, {X_horaire.shape[1]} features")

def evaluer(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n── {label} ──")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    return rf, X_train, X_test, y_train, y_test, y_pred, r2, mae, rmse

print("\n=== Entraînement Option 1 ===")
rf1, X_tr1, X_te1, y_tr1, y_te1, y_p1, r2_1, mae_1, rmse_1 = evaluer(X_exact, y_exact, "Option 1 — join exact")

print("\n=== Entraînement Option 2 ===")
rf2, X_tr2, X_te2, y_tr2, y_te2, y_p2, r2_2, mae_2, rmse_2 = evaluer(X_horaire, y_horaire, "Option 2 — agrégation horaire")

print("\n══════════════════════════════════════")
print("         RÉSUMÉ COMPARATIF")
print("══════════════════════════════════════")
print(f"{'':25} {'Option 1':>12} {'Option 2':>12}")
print(f"{'Nb lignes':25} {len(df_exact):>12} {len(df_horaire):>12}")
print(f"{'Nb features':25} {X_exact.shape[1]:>12} {X_horaire.shape[1]:>12}")
print(f"{'R²':25} {r2_1:>12.4f} {r2_2:>12.4f}")
print(f"{'MAE':25} {mae_1:>12.4f} {mae_2:>12.4f}")
print(f"{'RMSE':25} {rmse_1:>12.4f} {rmse_2:>12.4f}")
print("══════════════════════════════════════")

# ════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(y_te1, y_p1, alpha=0.3, s=12, color='#4C72B0')
lim = [-5, 105]
axes[0, 0].plot(lim, lim, 'r--', linewidth=1)
axes[0, 0].set(title=f'Option 1 — Réel vs Prédit\nR²={r2_1:.3f} | MAE={mae_1:.2f}',
               xlabel='Quality réelle', ylabel='Quality prédite')

axes[0, 1].hist(y_te1.values - y_p1, bins=40, color='#4C72B0', edgecolor='white')
axes[0, 1].axvline(0, color='red', linewidth=1.2, linestyle='--')
axes[0, 1].set(title='Option 1 — Résidus', xlabel='Résidu', ylabel='Fréquence')

axes[1, 0].scatter(y_te2, y_p2, alpha=0.3, s=12, color='#2CA02C')
axes[1, 0].plot(lim, lim, 'r--', linewidth=1)
axes[1, 0].set(title=f'Option 2 — Réel vs Prédit\nR²={r2_2:.3f} | MAE={mae_2:.2f}',
               xlabel='Quality réelle', ylabel='Quality prédite')

axes[1, 1].hist(y_te2.values - y_p2, bins=40, color='#2CA02C', edgecolor='white')
axes[1, 1].axvline(0, color='red', linewidth=1.2, linestyle='--')
axes[1, 1].set(title='Option 2 — Résidus', xlabel='Résidu', ylabel='Fréquence')

plt.suptitle('Comparaison : join exact vs agrégation horaire', fontsize=13)
plt.tight_layout()
plt.savefig("fig_comparaison_agregation.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] fig_comparaison_agregation.png")
