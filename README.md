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

En utilisant les H-1 et H-2 : R² = 0.9726, MAE  : 1.9815, RMSE : 2.7334

