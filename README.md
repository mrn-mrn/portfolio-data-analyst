# Portfolio sur l'analyse de données

## Projet 1 - Manipulation et pré-traitement de données

→ **Base de données complexe** sur les soutenances de thèses en France :  
- analyse et traitement des variables
- traitement des données manquantes (matrice de nullité, heatmap, dendrogramme)
- détection d’outliers et valeurs aberrantes 
- visualisations de données (barplots, lineplots)

→ **Rapport d’analyse** au format LaTeX.

→ **Outils** : Python (pandas, numpy, matplotlib, missingno, seaborn), Jupyter notebook.


## Projet 2 - Rédaction d'un rapport d'analyse

→ **Base de données** sur la perception du changement climatique en France :
- analyse et traitement des variables (variables dérivées, dataset intermédiaire)
- traitement des données manquantes (matrice de nullité)
- analyses statistiques descriptives avec tables et visualisations (barplots, heatmaps)
- visualisation détaillées (lineplot, ridgeplot, barplot avec décalage)

→ **Rapport d'analyse** au format IMRAD avec interprétations détaillées.

→ **Outils** : Python (pandas, numpy, matplotlib, seaborn), Jupyter Notebook, GitHub (versioning).


## Projet 3 - Machine Learning non supervisé

### a. Analyse en composantes principales

→ **Base de données** simple sur les données d'Iris :
- analyse et traitement des variables (centrage et réduction, sélection des variables pertinentes pour l’ACP)
- analyses statistiques descriptives avec tables et visualisations (corrélogrammes, scree plots, eigenvalues, factor loadings)
- visualisations détaillées (3D scatterplots, cercles des corrélations, biplots, représentation des contributions et qualités de représentation des individus et variables)

→ **Outils** : Python (pandas, numpy, matplotlib, seaborn, scikit-learn), Jupyter Notebook, GitHub (versioning).

### b. Réduction de dimensionnalité et clustering

→ **Base de données** sur les performances sportives :
- analyse et traitement des variables (standardisation, choix des variables pour ACP)
- traitement des données manquantes (vérification et exclusion des lignes incomplètes si nécessaire)
- analyses statistiques descriptives avec tables et visualisations (cercle des corrélations, scree plot, eigenvalues, factor loadings)
- visualisations détaillées (plan factoriel avec qualité de représentation, visualisation 3D des composantes principales, clusters sur le plan factoriel)

→ **Outils** : Python (pandas, numpy, matplotlib, seaborn, scikit-learn, scipy), Jupyter Notebook, GitHub (versioning).

### c. Analyse des correspondances multiples

→ **Base de données** sur des profils fictifs d’utilisateurs d’app de rencontre :
- analyse et traitement des variables (conversion des variables qualitatives en données pour ACM)
- traitement des données manquantes (éventuelles valeurs "non renseignées" traitées comme modalités spécifiques ou supprimées)
- analyses statistiques descriptives avec tables (inertie, contributions, cos² des variables)
- visualisations détaillées (plan factoriel des variables, représentation des profils-types émergents, analyse des contributions par axe)

→ **Outils** : Python (pandas, matplotlib, prince, seaborn), Jupyter Notebook, GitHub (versioning).

### d. Classification non supervisée avec DBSCAN

→ **Base de données** simulées de formes non convexes et lunes opposées :
- génération du jeu de données via make_moons() (ou données fournies par l’enseignant pour R)
- traitement préparatoire (normalisation si nécessaire, réduction dimensionnelle éventuelle)
- analyses comparatives des méthodes de clustering (k-means, CAH, DBSCAN)
- visualisations détaillées (graphiques de clustering, dendrogrammes 2D/3D, comparaison des partitions)

→ **Outils** : Python (pandas, scikit-learn, matplotlib, seaborn, scipy), Jupyter Notebook, GitHub (versioning).


## Projet 4 - Machine Learning supervisé

### a. Classification supervisée avec arbres de décision

→ **Base de données** sur la classification des espèces de manchots :
- apprentissage supervisé par arbres de décision (DecisionTreeClassifier)
- variation de la profondeur de l’arbre pour observer la complexité du modèle
- visualisation de la frontière de décision sur plan 2D (avec DecisionBoundaryDisplay)
- interprétation de la structure d’un arbre (plot_tree) : splits, feuilles, label majoritaire
- prédictions probabilistes sur de nouveaux points (predict_proba)
- analyse qualitative de l’impact des splits sur les classes
  
→ **Outils** : Python (scikit-learn, matplotlib, seaborn), Jupyter Notebook, GitHub (versioning).

### b. Classification binaire avec arbres de décision

→ **Base de données** sur les cas de cancer du sein :
- séparation aléatoire des données en ensembles d’entraînement et de test (train_test_split)
- entraînement d’arbres de décision avec réglage de la profondeur maximale (max_depth) et fixation d’un état aléatoire (random_state) pour la reproductibilité
- utilisation de différents critères d’impureté (gini, entropy) pour construire les arbres
- évaluation du modèle à l’aide de métriques classiques :
  * accuracy
  * matrice de confusion
  * précision
  * rappel
  * F1-score
  * courbes ROC et Precision-Recall
  * calcul des AUC correspondants
- visualisation des arbres de décision et des courbes de performance pour analyser les résultats
- interprétation des compromis entre métriques pour optimiser la classification binaire
  
→ **Outils** : Python (scikit-learn, matplotlib, seaborn), Jupyter Notebook, GitHub (versioning).

### c. Techniques avancées de régression avec arbres et forêts aléatoires

→ **Base de données** sur les véhicules et leur consommation d’essence :
- entraînement d’arbres de régression (DecisionTreeRegressor) avec réglage de la profondeur maximale et du nombre minimal d’échantillons par feuille
- séparation des données en ensembles d’entraînement et de test (80/20)
- calcul et interprétation des erreurs pour évaluer la qualité des prédictions :
  * MAE
  * MSE
  * RMSE 
- analyse de la performance sur les jeux d’entraînement et de test afin de détecter le sous-apprentissage ou surapprentissage
- construction et visualisation des courbes d’apprentissage pour étudier l’impact de la taille des données d’entraînement sur la performance
- introduction à la validation croisée (k-fold) pour obtenir une estimation robuste de l’erreur de généralisation
- comparaison entre arbres simples et forêts aléatoires (RandomForestRegressor), en comprenant le principe du bagging et la sélection aléatoire des variables à chaque split

→ **Outils** : Python (scikit-learn, matplotlib, seaborn), Jupyter Notebook, GitHub (versioning)

### d. Bagging, forêts aléatoires, validation croisée et tuning d’hyperparamètres

→ **Base de données** sur des diagnostics médicaux pour la détection de maladies du foie chez des patients.
- préparation des données (séparation train/test avec un ratio 70/30)
- mise en œuvre du bagging (BaggingClassifier) appliqué à des arbres de décision
- entraînement et évaluation de modèles avec des métriques classiques (accuracy, recall, AUC)
- comparaison entre bagging et forêts aléatoires (RandomForestClassifier) sur les mêmes données
- estimation et visualisation de l’importance des variables explicatives (feature importance) via des barplots
- optimisation des hyperparamètres par GridSearchCV pour améliorer les performances (tuning de max_depth, min_samples_leaf, etc.)
- analyse des résultats de la grille de recherche et sélection du meilleur modèle.

→ **Outils** : Python (sklearn ensemble, pandas, matplotlib, seaborn), Jupyter Notebook.

## Projet 5 - Techniques avancées de visualisation de données

→ **Base de données** sur les soutenances de thèses en France :
- analyse et traitement des variables qualitatives et quantitatives (discipline de rattachement, années, etc.)
- amélioration esthétique des graphiques (transparence, marges, polices, inclinaison des labels)
- visualisations statistiques descriptives classiques (stacked area plot, stacked bar chart)
- graphiques interactifs avancés avec sliders et selectors (Plotly)

→ **Outils** : Python (pandas, seaborn, matplotlib, plotly), Jupyter Notebook, GitHub (versioning).

## Projet 6 - Analyse statistique avancée

→ **Base de données** sur le MOOC 'Effectuation' :
- analyse et traitement des variables (fusion multi-sources, recodage, création de variables composites)
- traitement des données manquantes et harmonisation des formats
- analyses statistiques descriptives avec tables :
  * chi2
  * t-test
  * ANOVA
  * tests non-paramétriques
- visualisations synthétiques (boxplots, mosaic plots)
- visualisations détaillées (scatterplots avec régressions, forest plots d’odd-ratios, diagnostics de modèles)

→ **Outils** : Python (pandas, numpy, scipy, statsmodels, matplotlib, seaborn), Jupyter Notebook, GitHub (versioning).

→ **Rapport d’analyse** au format IMRAD avec interprétations détaillées.

## Projet 7 - Business intelligence

→ **Base de données** RH :
- Traitement des données
- Sélection et traitement des variables pertinentes
- visualisations dynamiques (barplots filtrables, graphiques diachroniques)
- carte géographique interactive
- Key Influencers

→ **Outils** : Power BI (DAX).

→ **Tableau de bord** interactif.
