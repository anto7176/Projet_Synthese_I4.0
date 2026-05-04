import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

# ════════════════════════════════════════════════════════════════════
# MODÈLE
# ════════════════════════════════════════════════════════════════════

df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

df_y_decale = df_y_final.copy()
df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)
df = df_x_final.join(df_y_decale, how='inner')

X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Modèle entraîné ✓")

# ════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ════════════════════════════════════════════════════════════════════

# Figure 1 : PDP des variables de la chambre 3
print("\nGénération PDP chambre 3...")
features_chambre3 = ['T_data_3_1', 'T_data_3_2', 'T_data_3_3']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, feature in zip(axes, features_chambre3):
    PartialDependenceDisplay.from_estimator(
        rf, X_train, [feature],
        ax=ax, line_kw={"color": "#D7263D", "linewidth": 2}
    )
    ax.set_title(f"PDP — {feature}", fontsize=12)
    ax.set_xlabel(feature)
    ax.set_ylabel("Effect sur quality")

plt.suptitle("Partial Dependence Plots — Chambre 3 (variables dominantes)", fontsize=13)
plt.tight_layout()
plt.savefig("fig_pdp_chambre3.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_pdp_chambre3.png")

# Figure 2 : PDP de toutes les variables
print("\nGénération PDP toutes variables...")
all_features = list(X.columns)

n_cols = 3
n_rows = int(np.ceil(len(all_features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i, feature in enumerate(all_features):
    PartialDependenceDisplay.from_estimator(
        rf, X_train, [feature],
        ax=axes[i], line_kw={"color": "#4C72B0", "linewidth": 2}
    )
    axes[i].set_title(feature, fontsize=10)
    axes[i].set_xlabel("")

for j in range(len(all_features), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Partial Dependence Plots — Toutes les variables", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("fig_pdp_toutes_variables.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_pdp_toutes_variables.png")

# Figure 3 : PDP 2D — interaction T_data_3_1 x H_data
print("\nGénération PDP 2D (interaction)...")
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    rf, X_train, [('T_data_3_1', 'H_data')],
    ax=ax, kind='average'
)
ax.set_title("PDP 2D — Interaction T_data_3_1 × H_data", fontsize=12)
plt.tight_layout()
plt.savefig("fig_pdp_2d_interaction.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_pdp_2d_interaction.png")
