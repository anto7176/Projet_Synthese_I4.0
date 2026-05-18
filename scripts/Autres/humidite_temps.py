import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Import'))
from import_data import charger_donnees, nettoyer_donnees, formater_index_temporel

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

df_x, _ = charger_donnees(
    os.path.join(BASE_DIR, "data/data_X.csv"),
    os.path.join(BASE_DIR, "data/data_Y.csv"),
)
df_x = nettoyer_donnees(df_x)
df_x = formater_index_temporel(df_x, "date_time")

# Agrégation horaire pour alléger l'affichage (2M points → ~35K points)
humidite_h = df_x['H_data'].resample('h').mean()

# Limites d'affichage robustes (ignore les outliers extrêmes)
y_low  = humidite_h.quantile(0.01)
y_high = humidite_h.quantile(0.99)
print(f"Plage affichée (p1–p99) : [{y_low:.1f}, {y_high:.1f}]  "
      f"(min réel : {humidite_h.min():.1f}, max réel : {humidite_h.max():.1f})")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
fig.suptitle("Humidité (H_data) au fil du temps", fontsize=13, fontweight='bold')

# --- Série horaire (échelle p1–p99) ---
axes[0].plot(humidite_h.index, humidite_h.values, color='#2E86AB', linewidth=0.6, alpha=0.8)
axes[0].set_ylim(y_low, y_high)
axes[0].set_ylabel("Humidité")
axes[0].set_title(f"Série horaire  (échelle p1–p99 : {y_low:.0f}–{y_high:.0f})")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha='right')

# --- Moyenne mensuelle ---
humidite_m = humidite_h.resample('ME').mean()
axes[1].bar(humidite_m.index, humidite_m.values, width=20, color='#2E86AB', alpha=0.7)
axes[1].set_ylim(y_low, y_high)
axes[1].set_ylabel("Humidité moyenne")
axes[1].set_title("Moyenne mensuelle")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
out = os.path.join(BASE_DIR, "figures", "humidite_temps.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"[Saved] {out}")

print(f"\nStatistiques H_data :")
print(humidite_h.describe().round(2))
