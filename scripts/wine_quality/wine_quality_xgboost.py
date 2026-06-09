import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

BASE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'wine_quality')

# ---- CHARGEMENT ----
df_red   = pd.read_csv(os.path.join(DATA_DIR, 'winequality-red.csv'),   sep=';')
df_white = pd.read_csv(os.path.join(DATA_DIR, 'winequality-white.csv'), sep=';')

df_red['type']   = 0
df_white['type'] = 1

df = pd.concat([df_red, df_white], ignore_index=True)

print(f"Dataset combiné : {len(df)} échantillons ({len(df_red)} rouges + {len(df_white)} blancs)")
print(f"Distribution qualité :\n{df['quality'].value_counts().sort_index().to_string()}\n")

FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol', 'type',
]

X = df[FEATURES].astype(float)
y = df['quality'].astype(int) - 3  # remap 3→9 en 0→6 (requis par XGBClassifier)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- MODELE ----
model = XGBClassifier(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0.1,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    tree_method      = 'hist',
    random_state     = 42,
    eval_metric      = 'mlogloss',
    num_class        = 7,
)

print("Entraînement en cours...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Remap inverse pour affichage
y_test_disp = y_test + 3
y_pred_disp = y_pred + 3

acc    = accuracy_score(y_test, y_pred)
# Accuracy ±1 : prédit correct si écart <= 1 note
acc_1  = np.mean(np.abs(y_pred - y_test) <= 1)

print(f"\nRésultats sur jeu de test :")
print(f"Accuracy exacte : {acc:.4f}")
print(f"Accuracy ±1     : {acc_1:.4f}")
print(f"\n{classification_report(y_test_disp, y_pred_disp)}")

# ---- AFFICHAGE ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"XGBoost Classifier — Wine Quality | Acc={acc:.4f} | Acc±1={acc_1:.4f}",
    fontsize=13, fontweight='bold'
)

# Matrice de confusion
cm = confusion_matrix(y_test_disp, y_pred_disp)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=range(3, 10), yticklabels=range(3, 10))
axes[0].set_xlabel("Prédit")
axes[0].set_ylabel("Réel")
axes[0].set_title("Matrice de confusion")

# Importance des features
feat_imp = pd.Series(model.feature_importances_, index=FEATURES)
feat_imp.sort_values(ascending=False).plot(kind='barh', ax=axes[1], color='#4C72B0')
axes[1].set_title("Importance des variables")
axes[1].set_xlabel("Importance")

# Distribution des erreurs (en nombre de notes d'écart)
errors = y_pred_disp - y_test_disp
axes[2].hist(errors, bins=range(-6, 8), color='#DD8452', edgecolor='white',
             linewidth=0.5, align='left')
axes[2].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[2].set_xlabel("Écart (prédit − réel)")
axes[2].set_ylabel("Nombre d'échantillons")
axes[2].set_title(f"Distribution des erreurs\nBiais moyen : {errors.mean():.3f}")
axes[2].set_xticks(range(-6, 7))

plt.tight_layout()
out = os.path.join(BASE_DIR, 'figures', 'wine_quality_xgboost_clf.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n[Saved] {out}")