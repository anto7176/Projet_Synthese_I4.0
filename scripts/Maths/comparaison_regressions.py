import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, PoissonRegressor, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from scipy.special import logit, expit

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

VARS = ['T_data_3_1', 'T_data_5_1', 'H_data', 'T_data_1_3', 'T_data_2_2', 'T_data_4_1']

df_t = df_x[VARS].resample('h').mean()
df_q = df_y[['quality']].copy()
df_q.index = df_q.index.floor('h')

df = df_q.join(df_t, how='inner').dropna()
print(f"Points alignés : {len(df)}")

X = df[VARS].values
y = df['quality'].values

# Régression linéaire
lin = LinearRegression().fit(X, y)
y_lin = lin.predict(X)
r2_lin = r2_score(y, y_lin)

terms = " + ".join(f"({c:.6f}x{v})" for c, v in zip(lin.coef_, VARS))
print(f"\n=== Linéaire  (R²={r2_lin:.4f}) ===")
print(f"qualité = {terms} + ({lin.intercept_:.4f})")

# Régression polynomiale degré 2
poly_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
poly_model.fit(X, y)
y_poly = poly_model.predict(X)
r2_poly = r2_score(y, y_poly)

feat_names = poly_model.named_steps['polynomialfeatures'].get_feature_names_out(VARS)
coefs = poly_model.named_steps['linearregression'].coef_
intercept_poly = poly_model.named_steps['linearregression'].intercept_
print(f"\n=== Polynomiale degré 2  (R²={r2_poly:.4f}) ===")
terms_poly = " + ".join(f"({c:.4f}x{n})" for c, n in zip(coefs, feat_names))
print(f"qualité = {terms_poly} + ({intercept_poly:.4f})")

# Régression exponentielle (log-linéaire) : quality = exp(a·X + b)
y_clipped = np.clip(y, 0.01, None)
log_y = np.log(y_clipped)

exp_lin = LinearRegression().fit(X, log_y)
y_exp = np.exp(exp_lin.predict(X))
r2_exp = r2_score(y, y_exp)

terms_exp = " + ".join(f"({c:.6f}x{v})" for c, v in zip(exp_lin.coef_, VARS))
print(f"\n=== Exponentielle  (R²={r2_exp:.4f}) ===")
print(f"qualité = exp({terms_exp} + ({exp_lin.intercept_:.4f}))")

# Régression de Poisson (requiert y > 0)
y_poisson = y + 1.0
poisson = PoissonRegressor(max_iter=500).fit(X, y_poisson)
y_pois = poisson.predict(X) - 1.0
r2_pois = r2_score(y, y_pois)

terms_pois = " + ".join(f"({c:.6f}x{v})" for c, v in zip(poisson.coef_, VARS))
print(f"\n=== Poisson  (R²={r2_pois:.4f}) ===")
print(f"qualité = exp({terms_pois} + ({poisson.intercept_:.4f})) - 1")

# S-Curve (logit-linéaire) : quality = 100 × sigmoid(a·X + b)
y_norm = np.clip(y / 100, 0.001, 0.999)
scurve_lin = LinearRegression().fit(X, logit(y_norm))
y_scurve = 100 * expit(scurve_lin.predict(X))
r2_scurve = r2_score(y, y_scurve)

terms_sc = " + ".join(f"({c:.6f}x{v})" for c, v in zip(scurve_lin.coef_, VARS))
print(f"\n=== S-Curve  (R²={r2_scurve:.4f}) ===")
print(f"qualité = 100 x sigmoid({terms_sc} + ({scurve_lin.intercept_:.4f}))")

# Lasso (régression linéaire + régularisation L1)
lasso = Lasso(alpha=0.1, max_iter=5000).fit(X, y)
y_lasso = lasso.predict(X)
r2_lasso = r2_score(y, y_lasso)

terms_lasso = " + ".join(f"({c:.6f}x{v})" for c, v in zip(lasso.coef_, VARS))
print(f"\n=== Lasso (α=0.1)  (R²={r2_lasso:.4f}) ===")
print(f"qualité = {terms_lasso} + ({lasso.intercept_:.4f})")
zero_coefs = [v for c, v in zip(lasso.coef_, VARS) if c == 0]
if zero_coefs:
    print(f"  Variables annulées par Lasso : {zero_coefs}")

# ---- AFFICHAGE ----
models_results = [
    ("Linéaire",          y_lin,    r2_lin,    '#4C72B0'),
    ("Polynomiale (d=2)", y_poly,   r2_poly,   '#55A868'),
    ("Exponentielle",     y_exp,    r2_exp,    '#DD8452'),
    ("Poisson",           y_pois,   r2_pois,   '#C44E52'),
    ("S-Curve",           y_scurve, r2_scurve, '#8172B2'),
    ("Lasso (α=0.1)",     y_lasso,  r2_lasso,  '#64B5CD'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Prédit vs Réel — 6 types de régression", fontsize=13, fontweight='bold')

for ax, (name, y_pred, r2, color) in zip(axes.flat, models_results):
    ax.scatter(y, y_pred, alpha=0.1, s=3, color=color)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, color='black', linewidth=1.2, label='Parfait')
    ax.set_xlabel("Qualité réelle")
    ax.set_ylabel("Qualité prédite")
    ax.set_title(f"{name}  (R²={r2:.3f})")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# S-Curve détaillée avec plage d'erreur
residuals = y_scurve - y
sigma = residuals.std()

order = np.argsort(y)
y_sorted = y[order]
y_pred_sorted = y_scurve[order]

window = 200
y_pred_smooth = np.convolve(y_pred_sorted, np.ones(window) / window, mode='valid')
y_real_smooth = np.convolve(y_sorted, np.ones(window) / window, mode='valid')

fig2, ax = plt.subplots(figsize=(9, 7))
ax.scatter(y, y_scurve, alpha=0.08, s=3, color='#8172B2', label='Points')
ax.plot(y_real_smooth, y_pred_smooth, color='#8172B2', linewidth=2, label='Courbe S-Curve')
ax.fill_between(y_real_smooth,
                y_pred_smooth - sigma,
                y_pred_smooth + sigma,
                alpha=0.25, color='#8172B2', label=f'±1σ ({sigma:.2f})')
lims = [min(y.min(), y_scurve.min()), max(y.max(), y_scurve.max())]
ax.plot(lims, lims, color='black', linewidth=1.2, linestyle='--', label='Parfait')
ax.set_xlabel("Qualité réelle")
ax.set_ylabel("Qualité prédite")
ax.set_title(f"S-Curve — Prédit vs Réel avec plage d'erreur  (R²={r2_scurve:.3f})")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

print("\n=== Récapitulatif R² ===")
for name, _, r2, _ in models_results:
    print(f"  {name:<22} R² = {r2:.4f}")
