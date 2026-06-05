import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Import'))
from import_data import charger_donnees, nettoyer_donnees, normaliser_qualite, formater_index_temporel

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

df_x, df_y = charger_donnees(
    os.path.join(BASE_DIR, "data/data_X.csv"),
    os.path.join(BASE_DIR, "data/data_Y.csv"),
)

df_x = nettoyer_donnees(df_x)
df_y = normaliser_qualite(df_y, colonne='quality')
df_x = formater_index_temporel(df_x, "date_time")
df_y = formater_index_temporel(df_y, "date_time")

cols_temp = [c for c in df_x.columns if c.startswith('T_')]
df_temp_h = df_x[cols_temp].resample('h').mean()
df_temp_h['temp_moy'] = df_temp_h.mean(axis=1)

# Les timestamps de data_Y ne tombent pas nécessairement à HH:00
df_y_h = df_y[['quality']].copy()
df_y_h.index = df_y_h.index.floor('h')

df = df_y_h.join(df_temp_h['temp_moy'], how='inner').dropna()
print(f"Données alignées : {len(df)} points ({df.index.min().date()} → {df.index.max().date()})")

df_mensuel = df.resample('ME').agg({'quality': 'mean', 'temp_moy': 'mean'})

r, p = stats.pearsonr(df['temp_moy'], df['quality'])
print(f"\nCorrélation de Pearson (quality vs temp_moy) : r = {r:.3f}  (p = {p:.2e})")

# ---- AFFICHAGE ----
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
fig.suptitle("Qualité et température des chambres au fil du temps", fontsize=14, fontweight='bold')

ax1 = axes[0]
ax1.plot(df.index, df['quality'], color='#4C72B0', linewidth=0.6, alpha=0.7)
ax1.set_ylabel("Qualité (normalisée 0–100)")
ax1.set_title("(a) Qualité horaire")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

ax2 = axes[1]
ax2.plot(df.index, df['temp_moy'], color='#DD8452', linewidth=0.6, alpha=0.7)
ax2.set_ylabel("Temp. moy. chambres (°C×10)")
ax2.set_title("(b) Température moyenne des 5 chambres (horaire)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

ax3 = axes[2]
ax3b = ax3.twinx()

ax3.bar(df_mensuel.index, df_mensuel['quality'],
        width=20, color='#4C72B0', alpha=0.6, label='Qualité moy. mensuelle')
ax3b.plot(df_mensuel.index, df_mensuel['temp_moy'],
          color='#DD8452', marker='o', markersize=4, linewidth=1.5, label='Temp. moy. mensuelle')

ax3.set_ylabel("Qualité moyenne mensuelle", color='#4C72B0')
ax3b.set_ylabel("Température moyenne mensuelle", color='#DD8452')
ax3.set_title(f"(c) Comparaison mensuelle  —  r = {r:.3f}")
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.tight_layout()
out1 = os.path.join(BASE_DIR, "figures", "qualite_vs_temperature_series.png")
os.makedirs(os.path.dirname(out1), exist_ok=True)
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.show()
print(f"[Saved] {out1}")

fig2, ax = plt.subplots(figsize=(7, 6))

mois = df.index.month
scatter = ax.scatter(df['temp_moy'], df['quality'],
                     c=mois, cmap='hsv', alpha=0.15, s=2, linewidths=0)
cb = plt.colorbar(scatter, ax=ax, ticks=range(1, 13))
cb.set_label("Mois")
cb.set_ticklabels(['Jan','Fév','Mar','Avr','Mai','Jun',
                   'Jul','Aoû','Sep','Oct','Nov','Déc'])

slope, intercept, *_ = stats.linregress(df['temp_moy'], df['quality'])
x_line = np.linspace(df['temp_moy'].min(), df['temp_moy'].max(), 200)
ax.plot(x_line, slope * x_line + intercept, color='black', linewidth=1.5,
        label=f'Régression  (r = {r:.3f})')

ax.set_xlabel("Température moyenne des chambres")
ax.set_ylabel("Qualité (normalisée 0–100)")
ax.set_title("Nuage de points : Qualité vs Température\n(couleur = mois)")
ax.legend(fontsize=9)

plt.tight_layout()
out2 = os.path.join(BASE_DIR, "figures", "qualite_vs_temperature_scatter.png")
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.show()
print(f"[Saved] {out2}")

def saison(m):
    if m in (12, 1, 2):  return 'Hiver'
    if m in (3, 4, 5):   return 'Printemps'
    if m in (6, 7, 8):   return 'Été'
    return 'Automne'

df['saison'] = df.index.month.map(saison)
print("\nMoyennes par saison :")
print(df.groupby('saison')[['quality', 'temp_moy']].mean().round(2))
print("\nCorrélation quality/temp_moy par saison :")
for s, g in df.groupby('saison'):
    rc, _ = stats.pearsonr(g['temp_moy'], g['quality'])
    print(f"  {s:<12} r = {rc:.3f}  (n = {len(g)})")
