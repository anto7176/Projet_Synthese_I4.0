# Projet\_Synthese\_I4.0





Essayer de trouver comment la variable "Quality" est calculée

Regarder si une variable peut être enlever pour l'aspect écologique (diagramme de Pareto)



R²   : 0.9263

MAE  : 3.1831

RMSE : 4.4452



R²   : 0.9598

MAE  : 2.3060

RMSE : 3.2835



Corrélation quality(t) vs quality(t-n) :

&#x20; h -  1h : 0.964

&#x20; h -  2h : 0.895

&#x20; h -  3h : 0.809

&#x20; h -  6h : 0.545

&#x20; h - 12h : 0.216

&#x20; h - 24h : 0.025

Pouvons nous utiliser les résultats de Qualité des heures précédentes pour estimée celle à un instant T ?

Si oui, il faut faire l'hypothèse que les mesures sont réalisées directement après chaque fin de cycle et que a H0, le résultat de la qualité H-1 est déjà disponible.



En utilisant les H-1 et H-2 : R² = 0.9726, MAE  : 1.9815, RMSE : 2.7334 avec ces param :

model =

XGBRegressor(

&#x20;   n\_estimators=3000,

&#x20;   learning\_rate=0.01,

&#x20;   max\_depth=7,

&#x20;   subsample=0.7,

&#x20;   colsample\_bytree=0.3,

&#x20;   min\_child\_weight=3,

&#x20;   early\_stopping\_rounds=50,

&#x20;   random\_state=42,

&#x20;   verbosity=0,

&#x20;   tree\_method='hist',

&#x20;   device='cuda'

)



faire intervalle pour le XGBoost

essayer de trouver pk sa montre, pk sa baisse

voir si lien avec les mois de l'année(temp,humidite)











hyperparametre : MEILLEURS PARAMÈTRES  (RMSE val = 2.3693)

═══════════════════════════════════════════════════════

&#x20; n\_lags          : 2

&#x20; n\_estimators    : 1306  (via early stopping)

&#x20; learning\_rate         : 0.0214113

&#x20; max\_depth             : 6

&#x20; subsample             : 0.586508

&#x20; colsample\_bytree      : 0.816817

&#x20; min\_child\_weight      : 4

&#x20; gamma                 : 2.41974

&#x20; reg\_alpha             : 3.35611e-05

&#x20; reg\_lambda            : 4.34695e-08

═══════════════════════════════════════════════════════



Entraînement du modèle final (train + val)...



── Résultats sur jeu de test ──

R²   : 0.9755

MAE  : 1.9168 = 5.09 échelle réelle

RMSE : 2.5842





Pour le poster, enlever la temp pour les valeurs cles et la emttre a la fin pour percuter

cahnger la problematique en mode : comment ameliorer une production de café

rajouter des grains de café en image



