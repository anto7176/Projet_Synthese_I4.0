import sys, os
import pandas as pd
import matplotlib.pyplot as plt

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

df_t = df_x[['T_data_3_1', 'T_data_5_1']].resample('h').mean()
df_q = df_y[['quality']].copy()
df_q.index = df_q.index.floor('h')

df = df_q.join(df_t, how='inner').dropna()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(df['T_data_3_1'], df['T_data_5_1'], df['quality'],
                c=df['quality'], cmap='viridis', alpha=0.2, s=3)
plt.colorbar(sc, ax=ax, label='Qualité (0–100)', shrink=0.5)

ax.set_xlabel("T_data_3_1")
ax.set_ylabel("T_data_5_1")
ax.set_zlabel("Qualité (0–100)")
ax.set_title("Qualité vs T_data_3_1 vs T_data_5_1")

plt.tight_layout()
plt.show()
