# Portfolio sur l'analyse de données

Bienvenue sur mon portfolio d’analyse de données.
Vous y trouverez plusieurs projets couvrant le spectre de la data analyse, du pré-traitement au machine learning supervisé/non supervisé, en passant par la data visualisation et la business intelligence.
Chaque projet contient les étapes clés, les outils utilisés et les résultats obtenus.

[Projet 1 - Manipulation et pré-traitement de données](#projet-1---manipulation-et-pré-traitement-de-données)  
[Projet 2 - Rédaction d'un rapport d'analyse](#projet-2---rédaction-dun-rapport-danalyse)  
[Projet 3 - Machine Learning non supervisé](#projet-3---machine-learning-non-supervisé)  
    [a. Analyse en composantes principales (ACP)](#a-analyse-en-composantes-principales-acp)  
    [b. Réduction de dimensionnalité et clustering](#b-réduction-de-dimensionnalité-et-clustering)  
    [c. Analyse des correspondances multiples (ACM)](#c-analyse-des-correspondances-multiples-acm)  
    [d. Classification non supervisée avec DBSCAN](#d-classification-non-supervisée-avec-dbscan)  
[Projet 4 - Machine Learning supervisé](#projet-4---machine-learning-supervisé)  
    [a. Classification supervisée avec arbres de décision](#a-classification-supervisée-avec-arbres-de-décision)  
    [b. Classification binaire avec arbres de décision](#b-classification-binaire-avec-arbres-de-décision)  
    [c. Techniques avancées de régression avec arbres et forêts aléatoires](#c-techniques-avancées-de-régression-avec-arbres-et-forêts-aléatoires)  
    [d. Bagging, forêts aléatoires, validation croisée et tuning d’hyperparamètres](#d-bagging-forêts-aléatoires-validation-croisée-et-tuning-dhyperparamètres)  
[Projet 5 - Techniques avancées de visualisation de données](#projet-5---techniques-avancées-de-visualisation-de-données)  
[Projet 6 - Analyse statistique avancée](#projet-6---analyse-statistique-avancée)  
[Projet 7 - Business intelligence](#projet-7---business-intelligence)


## Projet 1 - Manipulation et pré-traitement de données

→ **Base de données complexe** sur les soutenances de thèses en France :  
- analyse et traitement des variables
- traitement des données manquantes :
  * matrice de nullité (`missingno.matrix`)
  * carte thermique des données manquantes (`missingno.heatmap`)
  * dendrogramme des données manquantes (`missingno.dendrogram`)
- détection des valeurs aberrantes / outliers :
  * visualisation détaillée (`FacetGrid`)
  * table intermédiaire avec filtres logiques
- visualisations de données (`lineplot`,`barplot`)
    
→ **Rapport d’analyse** avec statistiques descriptives, en $LaTeX$.

→ **Outils** : `Python` (`pandas`, `matplotlib`, `missingno`, `seaborn`), `Jupyter notebook`.

→ **Résultat** : Rapport structuré contenant une sélection commentée de résultats et figures, avec analyse des outliers, des patterns et premières interprétations.


## Projet 2 - Rédaction d'un rapport d'analyse

→ **Base de données** sur la perception du changement climatique en France :
- analyse et traitement des variables :
  * variables dérivées
  * fusion de sous-ensembles (`groupby`, `merge`)
  * calculs statistiques de base
  * export de la table intermédiaire au format `.csv`
- traitement des données manquantes
- calculs statistiques de base
- figures travaillées (`lineplot`, `ridgeplot`, `barplot`)

→ **Rapport d'analyse** suivant la structure IMRaD avec interprétations détaillées.

→ **Outils** : `Python` (`pandas`, `numpy`, `matplotlib`, `seaborn`, `missingno`), `Jupyter Notebook`, `GitHub`.

→ **Résultat** : Présentation rigoureuse de résultats inédits issus d’une analyse personnelle de données, avec interprétations approfondies.


## Projet 3 - Machine Learning non supervisé

### a. Analyse en composantes principales (ACP)

→ **Base de données** simple sur les données d'Iris :
- analyse, traitement et visualisation des variables :
   * centrage et réduction des données
   * visualisation des données centrées-réduites (`scatterplot 3D`)
   * mise en oeuvre de l'ACP
   * représentation graphique des données après ACP
- analyse et visualisation des corrélations entre variables :
   * corrélogramme
   * cercle des corrélations
   * biplot des composantes principales
   * scree plot
   * table des valeurs propres (`eigenvalues`) et des saturations (`factor loadings`)
- analyse de la qualité de représentation des variables et des individus :  
  * Cos²
  * contributions
 - mise en oeuvre de l'algorithme k-means sur les composantes principales :
   * visualisation des clusters
   * détermination du nombre optimal de clusters (méthode du coude, des silhouettes)

→ **Outils** : `Python` (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `yellowbrick`, `prince`, `psynlig`), `Jupyter Notebook`, `GitHub`.

→ **Résultat** : la réduction de dimensionnalité condense l’information de plusieurs variables corrélées en un petit nombre de composantes principales indépendantes, ce qui facilite la réalisation du clustering. Le clustering permet alors de regrouper les iris en différentes espèces selon leurs caractéristiques.


### b. Réduction de dimensionnalité et clustering

→ **Base de données** sur les performances sportives :
- analyse et traitement des variables :
   * standardisation
   * choix des variables pour ACP
- traitement des données manquantes
- analyses statistiques descriptives avec tables et visualisations :
   * cercle des corrélations
   * `scree plot`
   * eigenvalues
   * factor loadings
- visualisations détaillées :
   * plan factoriel avec qualité de représentation
   * visualisation 3D des composantes principales
   * clusters sur le plan factoriel

→ **Outils** : `Python` (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`), `Jupyter Notebook`, `GitHub` (gestion des versions).

→ **Résultat** : 


### c. Analyse des correspondances multiples (ACM)

→ **Base de données** sur des profils fictifs d’utilisateurs d’application de rencontre :
- analyse et traitement des variables :
   * conversion des variables qualitatives en données pour ACM
- traitement des données manquantes
- analyses statistiques descriptives avec tables :
   * inertie
   * contributions
   * cos² des variables
- visualisations détaillées :
   * plan factoriel des variables
   * représentation des profils-types émergents
   * analyse des contributions par axe

→ **Outils** : `Python` (`pandas`, `matplotlib`, `prince`, `seaborn`), `Jupyter Notebook`, `GitHub` (gestion des versions).

→ **Résultat** : 


### d. Classification non supervisée avec DBSCAN

→ **Base de données** simulées de formes non convexes et lunes opposées :
- génération du jeu de données (`make_moons()`)
- traitement préparatoire
- analyses comparatives des méthodes de clustering :
   * k-means
   * CAH
   * DBSCAN
- visualisations détaillées :
   * graphiques de clustering
   * dendrogrammes 2D/3D
   * comparaison des partitions

→ **Outils** : `Python` (`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`), `Jupyter Notebook`, `GitHub` (gestion des versions).

→ **Résultat** : 


## Projet 4 - Machine Learning supervisé

### a. Classification supervisée avec arbres de décision

→ **Base de données** sur la classification des espèces de manchots :
- apprentissage supervisé par arbres de décision (`DecisionTreeClassifier`)
- variation de la profondeur de l’arbre pour observer la complexité du modèle
- visualisation de la frontière de décision sur plan 2D (`DecisionBoundaryDisplay`)
- interprétation de la structure d’un arbre (`plot_tree`) :
   * splits
   * feuilles
   * label majoritaire
- prédictions probabilistes sur de nouveaux points (`predict_proba`)
- analyse qualitative de l’impact des splits sur les classes
  
→ **Outils** : `Python` (`scikit-learn`, `matplotlib`, `seaborn`), `Jupyter Notebook`, `GitHub` (gestion des versions).

→ **Résultat** : 


### b. Classification binaire avec arbres de décision

→ **Base de données** sur les cas de cancer du sein :
- séparation aléatoire des données en ensembles d’entraînement et de test (`train_test_split`)
- entraînement d’arbres de décision :
    * réglage de la profondeur maximale (`max_depth`)
    * fixation d’un état aléatoire (`random_state`) pour la reproductibilité
- utilisation de différents critères d’impureté pour construire les arbres :
  * indice de gini
  * entropy
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
  
→ **Outils** : `Python` (`scikit-learn`, `matplotlib`, `seaborn`), `Jupyter Notebook`, `GitHub` (gestion des versions).

→ **Résultat** : 


### c. Techniques avancées de régression avec arbres et forêts aléatoires

→ **Base de données** sur les véhicules et leur consommation d’essence :
- entraînement d’arbres de régression (`DecisionTreeRegressor`) :
   * réglage de la profondeur maximale
   * réglage du nombre minimal d’échantillons par feuille
- séparation des données en ensembles d’entraînement et de test (80/20)
- calcul et interprétation des erreurs pour évaluer la qualité des prédictions :
  * MAE
  * MSE
  * RMSE 
- analyse de la performance sur les jeux d’entraînement et de test :
   * détection du sous-apprentissage 
   * détection du surapprentissage
- analyse de l’impact de la taille des données d’entraînement sur la performance
   * courbes d’apprentissage
- estimation robuste de l’erreur de généralisation :
   * validation croisée (`k-fold`)
- comparaison entre arbres simples et forêts aléatoires (`RandomForestRegressor`)

→ **Outils** : `Python` (`scikit-learn`, `matplotlib`, `seaborn`), `Jupyter Notebook`, `GitHub` (gestion des versions)

→ **Résultat** : 


### d. Bagging, forêts aléatoires, validation croisée et tuning d’hyperparamètres

→ **Base de données** sur la détection de maladies du foie chez des patients :
- préparation des données :
   * séparation train/test (ratio 70/30)
- bagging (`BaggingClassifier`) appliqué à des arbres de décision
- entraînement du modèle
- évaluation du modèle avec des métriques classiques :
   * accuracy
   * recall
   * AUC
- comparaison entre bagging et forêts aléatoires (`RandomForestClassifier`)
- estimation de l'importance des variables explicatives (`barplot`)
- optimisation des hyperparamètres (`GridSearchCV`) :
   * tuning de `max_depth`
   * tuning de `min_samples_leaf`
- analyse des résultats et sélection du meilleur modèle.

→ **Outils** : `Python` (`sklearn`, `pandas`, `matplotlib`, `seaborn`), `Jupyter Notebook`.

→ **Résultat** : 


## Projet 5 - Techniques avancées de visualisation de données

→ **Base de données** sur les soutenances de thèses en France :
- analyse et traitement des variables qualitatives et quantitatives (discipline de rattachement, années, etc.)
- amélioration esthétique des graphiques (transparence, marges, polices, inclinaison des labels)
- visualisations statistiques descriptives classiques (stacked area plot, stacked bar chart)
- graphiques interactifs avancés avec sliders et selectors (Plotly)

→ **Outils** : Python (pandas, seaborn, matplotlib, plotly), Jupyter Notebook, GitHub (versioning).

→ **Résultat** : 


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

→ **Résultat** : 


## Projet 7 - Business intelligence

→ **Base de données** RH :
- Traitement des données
- Sélection et traitement des variables pertinentes
- visualisations dynamiques (barplots filtrables, graphiques diachroniques)
- carte géographique interactive
- Key Influencers

→ **Outils** : Power BI (DAX).

→ **Tableau de bord** interactif.

→ **Résultat** : 
