import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'wine_quality')

df_red   = pd.read_csv(os.path.join(DATA_DIR, 'winequality-red.csv'),   sep=';')
df_white = pd.read_csv(os.path.join(DATA_DIR, 'winequality-white.csv'), sep=';')
df_red['type'], df_white['type'] = 0, 1
df = pd.concat([df_red, df_white], ignore_index=True)

FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol',
]

# On prend les meilleurs vins (note >= 7) et on regarde leur valeur typique
bons = df[df['quality'] >= 8.5]

print("VALEUR IDÉALE PAR VARIABLE (pour viser une bonne qualité)")
print("=" * 55)
for feat in FEATURES:
    print(f"{feat:<22} : {bons[feat].median():.3f}")
print("=" * 55)
print(f"(basé sur {len(bons)} vins notés >= 7)")