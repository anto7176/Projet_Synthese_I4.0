# Torréfaction intelligente : optimisation de la qualité de production (Industrie 4.0)

Projet de synthèse réalisé en binôme (4 semaines) dans le cadre de la spécialisation Industrie 4.0 à l'ESEO.

## Contexte

La torréfaction du café est un procédé thermique réparti sur **5 chambres** et instrumenté de **17 capteurs**. La qualité du produit est mesurée en sortie de machine.

**Problématique :** comment optimiser la qualité d'une production de café torréfié à partir des données capteurs ?

## Démarche

1. **Exploration et nettoyage** des données de production issues des capteurs.
2. **Analyse des corrélations** entre les variables du procédé et la qualité finale.
3. **Sélection de variables** via un diagramme de Pareto, avec une réflexion sur la réduction du nombre de capteurs utiles (enjeu écologique et économique).
4. **Comparaison de plusieurs modèles** de prédiction (Random Forest, XGBoost et autres), ainsi qu'une approche par modèle de langage (LLM).
5. **Optimisation** des hyperparamètres et introduction de variables retardées pour affiner les prédictions.

Le modèle **XGBoost** s'est révélé le plus performant et a été retenu.

## Approche par variables retardées

Une piste explorée : utiliser la qualité mesurée aux heures précédentes pour estimer la qualité à l'instant T.

Corrélation entre la qualité à l'instant T et aux instants antérieurs :

| Décalage | Corrélation |
|----------|-------------|
| T − 1 h  | 0.964 |
| T − 2 h  | 0.895 |
| T − 3 h  | 0.809 |
| T − 6 h  | 0.545 |
| T − 12 h | 0.216 |
| T − 24 h | 0.025 |

La corrélation reste très forte sur les 1 à 3 dernières heures, ce qui justifie l'ajout des valeurs T−1 et T−2 comme variables d'entrée.

> Hypothèse retenue : les mesures sont réalisées juste après chaque fin de cycle, et à l'instant T0 le résultat de qualité de T−1 est déjà disponible.

## Résultats

Progression des performances au fil des itérations :

| Version | R² | MAE | RMSE |
|---------|-----|-----|------|
| Modèle initial | 0.926 | 3.18 | 4.45 |
| Modèle optimisé | 0.960 | 2.31 | 3.28 |
| + variables retardées (T−1, T−2) | 0.973 | 1.98 | 2.73 |
| **Modèle final (jeu de test)** | **0.976** | **1.92** | **2.58** |

Le modèle final atteint un **R² de 0,976**, soit une erreur moyenne d'environ **5 sur l'échelle réelle de qualité**. Il est exploitable comme outil d'aide à la décision pour piloter la qualité en production.

## Modèle final

```python
XGBRegressor(
    n_estimators=1306,        # via early stopping
    learning_rate=0.0214,
    max_depth=6,
    subsample=0.5865,
    colsample_bytree=0.8168,
    min_child_weight=4,
    gamma=2.4197,
    reg_alpha=3.36e-05,
    reg_lambda=4.35e-08,
    tree_method='hist',
    device='cuda',
    random_state=42,
)
# n_lags = 2 (variables retardées T-1 et T-2)
```

## Technologies

Python · XGBoost · scikit-learn · pandas · Machine Learning · Science des données

## Contenu du dépôt

- `scripts/` : chaîne de traitement et scripts de modélisation
- `figures/` : graphiques et visualisations
- `Poster_ProjetI4.0.pdf` : poster de présentation du projet
- `Presentation_ProjetDeSythese_I4_0.pdf` : présentation technique
- `Planning_previsionnel.xlsx` : planning prévisionnel du projet

## Pistes d'amélioration

- Ajouter un intervalle de confiance aux prédictions XGBoost.
- Analyser les causes des hausses et baisses de qualité dans le temps.
- Étudier le lien éventuel avec la saisonnalité (température, humidité selon les mois).
