import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

# ════════════════════════════════════════════════════════════════════
# MODÈLE
# ════════════════════════════════════════════════════════════════════

df_x, df_y = charger_donnees("data/data_X.csv", "data/data_Y.csv")

df_x_final = formater_index_temporel(nettoyer_donnees(df_x), "date_time")
df_y_final = formater_index_temporel(normaliser_qualite(df_y, colonne='quality'), "date_time")

# Décalage de Y d'1 heure en arrière avant la fusion
df_y_decale = df_y_final.copy()
df_y_decale.index = df_y_decale.index - pd.Timedelta(hours=1)

df = df_x_final.copy()
df['quality'] = df_y_decale['quality']

corr = df.corr()

corr_quality = corr['quality'].drop('quality').sort_values()

print("\n── Top 10 variables les plus corrélées avec quality ──")
print(corr_quality.abs().sort_values(ascending=False).head(10).to_string())

# ════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ════════════════════════════════════════════════════════════════════

# Figure 1 : Heatmap complète
fig, ax = plt.subplots(figsize=(18, 14))

sns.heatmap(
    corr,
    annot=False,
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.3,
    linecolor='white',
    ax=ax
)

ax.set_title("Matrice de corrélation — toutes les variables", fontsize=14, pad=15)
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)

plt.tight_layout()
plt.savefig("fig_correlation_complete.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_correlation_complete.png")

# Figure 2 : Corrélation de chaque variable avec quality
fig, ax = plt.subplots(figsize=(10, max(5, len(corr_quality) * 0.28)))

colors = ['#D7263D' if v > 0 else '#4C72B0' for v in corr_quality]
ax.barh(corr_quality.index, corr_quality.values, color=colors, edgecolor='none', height=0.7)

ax.axvline(0, color='black', linewidth=0.8)
ax.axvline( 0.3, color='gray', linewidth=0.6, linestyle='--', alpha=0.5)
ax.axvline(-0.3, color='gray', linewidth=0.6, linestyle='--', alpha=0.5)

ax.set_xlabel("Coefficient de corrélation de Pearson", fontsize=11)
ax.set_title("Corrélation de chaque variable avec quality", fontsize=13, pad=12)
ax.text(0.31,  0.01, "seuil +0.3", transform=ax.get_xaxis_transform(), fontsize=8, color='gray')
ax.text(-0.45, 0.01, "seuil -0.3", transform=ax.get_xaxis_transform(), fontsize=8, color='gray')

ax.legend(handles=[
    Patch(color='#D7263D', label='Corrélation positive'),
    Patch(color='#4C72B0', label='Corrélation négative'),
], fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig("fig_correlation_quality.png", dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] fig_correlation_quality.png")
