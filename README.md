# Portfolio sur l'analyse de données

Bienvenue sur mon portfolio dédié à l’analyse de données. Vous y découvrirez des projets variés allant du traitement et préparation des données à la prédiction par machine learning supervisé et non supervisé, en passant par la création de visualisations interactives, l’analyse statistique avancée, et la conception de tableaux de bord.

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


## [Projet 1](./projet1/) - Manipulation et pré-traitement de données

_Création d'un rapport structuré contenant une sélection commentée de résultats et graphiques, avec analyse des valeurs aberrantes, des tendances et premières interprétations._

→ **Jeu de données complexe** sur les soutenances de thèses en France :  
- analyse et traitement des variables
- traitement des données manquantes :
  * matrice de nullité
  * carte thermique des données manquantes
  * dendrogramme des données manquantes
- détection des valeurs aberrantes :
  * visualisation détaillée
  * table intermédiaire avec filtres logiques
- visualisations de données
    
→ **Rapport d’analyse** avec statistiques descriptives, en $LaTeX$.

→ **Outils** : `Python` (`pandas`, `matplotlib`, `missingno`, `seaborn`), `Jupyter Notebook`.


## [Projet 2](./projet2/) - Rédaction d'un rapport d'analyse

_Présentation selon les standards scientifiques de résultats inédits issus d’une analyse personnelle de données, avec interprétations approfondies._

→ **Jeu de données** sur la perception du changement climatique en France :
- analyse et traitement des variables :
  * variables dérivées
  * fusion de sous-ensembles
  * export de la table intermédiaire au format `.csv`
- traitement des données manquantes
- calculs statistiques de base
- figures travaillées

→ **Rapport d'analyse** suivant la structure IMRaD avec interprétations détaillées, en $LaTeX$.

→ **Outils** : `Python` (`pandas`, `numpy`, `matplotlib`, `seaborn`, `missingno`), `Jupyter Notebook`, `GitHub`.


## Projet 3 - Machine Learning non supervisé

### a. Analyse en composantes principales (ACP)

_Condensation de l’information de plusieurs variables corrélées en un petit nombre de composantes principales indépendantes, ce qui facilite la réalisation du clustering. Le clustering permet alors de regrouper les iris selon leurs caractéristiques._

→ **Jeu de données** simple sur les données d'Iris :
- analyse, traitement et visualisation des variables :
   * centrage et réduction des données
   * visualisation des données centrées-réduites
   * mise en oeuvre de l'ACP
   * représentation graphique des données après ACP
- analyse et visualisation des corrélations entre variables :
   * corrélogramme
   * cercle des corrélations
   * biplot des composantes principales
   * graphe de l'éboulis (scree plot)
   * table des valeurs propres et des saturations
- analyse de la qualité de représentation des variables et des individus :  
  * Cos²
  * contributions
 - mise en œuvre de l'algorithme k-means sur les composantes principales :
   * visualisation des clusters
   * détermination du nombre optimal de clusters (méthode du coude, des silhouettes)

→ **Outils** : `Python` (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `yellowbrick`), `Jupyter Notebook`, `GitHub`.


### b. Réduction de dimensionnalité et clustering

_Analyse des performances des athlètes en décathlon en regroupant les épreuves selon des qualités clés (vitesse, force, etc.) et comparaison de deux méthodes de clustering : k-means (nombre de clusters prédéfini), et la classification ascendante hiérarchique (CAH), qui construit une hiérarchie de groupes. Cette comparaison permet de mieux comprendre et visualiser les profils des sportifs._

→ **Jeu de données** sur les performances sportives :
- analyse et traitement des variables :
   * centrage et réduction des données
   * mise en oeuvre de l'ACP
   * cercle des corrélations
- représentation des individus sur le plan factoriel :
   * qualité de représentation des individus
   * table des contributions vectorielles
   * mise en oeuvre de l'algorithme k-means
   * mise en oeuvre de la méthode de CAH

→ **Outils** : `Python` (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`), `Jupyter Notebook`, `GitHub`.


### c. Analyse des correspondances multiples (ACM)

_Identification de profils-types d’utilisateurs d’une application de rencontre à partir de leurs caractéristiques qualitatives, ainsi que des variables les plus contributives._

→ **Jeu de données** sur des profils fictifs d’utilisateurs d’application de rencontre :
- analyse, traitement et représentation des variables :
   * nettoyage et conversion
   * mise en oeuvre de l'ACM
   * représentation des variables
- analyse de la qualité de représentation des variables :
   * inertie
   * contributions
   * Cos²
 
→ **Outils** : `Python` (`pandas`, `numpy`,`matplotlib`, `prince`), `Jupyter Notebook`, `GitHub`.


### d. Classification non supervisée avec DBSCAN

_Démonstration que DBSCAN est la méthode de clustering la plus adaptée pour détecter des structures non convexes, en comparant ses résultats à ceux de k-means et CAH sur un jeu de données en forme de lunes._

→ **Jeu de données** simulées :
- génération du jeu de données
- analyses comparatives des méthodes de clustering :
   * k-means
   * CAH
   * DBSCAN
- visualisations détaillées :
   * graphiques de clustering
   * comparaison des partitions

→ **Outils** : `Python` (`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`), `Jupyter Notebook`, `GitHub`.


## Projet 4 - Machine Learning supervisé

### a. Classification supervisée avec arbres de décision

_Prédiction de l’espèce d’un manchot (Adélie, Chinstrap ou Gentoo) à partir de ses caractéristiques morphologiques._

→ **Jeu de données** sur la classification des espèces de manchots :
- préparation des données :
   * analyse exploratoire et traitement des variables
   * séparation des caractéristiques et de leurs étiquettes
   * création de deux sous-ensembles d’entraînement et de test
- construction du modèle :
   * entraînement de l'arbre de classification
   * visualisation de la partition des données selon l'arbre
   * étude de l’impact de la profondeur de l’arbre sur les performances
- prédictions probabilistes pour un nouvel échantillon
  
→ **Outils** : `Python` (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`), `Jupyter Notebook`, `GitHub`.


### b. Classification binaire avec arbres de décision

_Prédiction de la nature bénigne ou maligne d’une tumeur du sein à partir de caractéristiques médicales, en utilisant la méthode du hold-out. Cette méthode divise les données en ensembles d’entraînement et de test afin d’évaluer la performance du modèle sur des données inédites, limitant ainsi le risque de surapprentissage (overfitting)._

→ **Jeu de données** sur les cas de cancer du sein :
- analyse et préparation des variables :
   * nettoyage des données
   * détection et traitement des valeurs aberrantes
   * matrice des corrélations
   * division aléatoire des données en ensembles d’entraînement et de test 
- construction et entraînement du modèle :
   * instanciation de l'arbre de décision
   * entraînement du modèle sur les données d'entraînement
- expérimentation avec différents critères d’impureté :
  * indice de gini
  * entropie
- évaluation du modèle à l’aide de métriques classiques :
  * accuracy
  * matrice de confusion
  * précision
  * rappel
  * F1-score
  * courbes ROC et Precision-Recall
  * calcul des AUC associés
- visualisation et analyse des résultats :
   * représentation graphique des arbres de décision
   * affichage des courbes de performance
   * interprétation des compromis entre métriques pour optimiser la classification
  
→ **Outils** : `Python` (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`), `Jupyter Notebook`, `GitHub`.


### c. Comparaison des modèles avec arbres et forêts aléatoires

_Prédiction de la consommation d’essence d’un véhicule (km/litre) à partir de ses caractéristiques, et comparaison de la performance entre un arbre de régression et une forêt aléatoire._

→ **Jeu de données** sur les véhicules et leur consommation d’essence :
- analyse et préparation des variables :
   * nettoyage des données
   * détection et traitement des valeurs aberrantes
   * matrice des corrélations
   * division en ensembles entraînement et test
- construction et entraînement de modèles de régression :
   * arbre de décision avec réglage des hyperparamètres
   * forêt aléatoire (bagging + sélection aléatoire de variables à chaque split)
- évaluation et comparaison des performances des modèles :
   * calcul des erreurs MAE, MSE, RMSE
   * analyse de l’erreur sur les jeux d'entraînement/test
   * courbes d’apprentissage pour identifier sous- ou surapprentissage
   * validation croisée pour estimer l’erreur de généralisation de manière plus robuste
- sélection du meilleur modèle sur la base des métriques de performance

→ **Outils** : `Python` (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`), `Jupyter Notebook`, `GitHub`.


### d. Bagging, forêts aléatoires et tuning d’hyperparamètres

_Prédiction de maladie du foie à partir de données médicales en comparant bagging et forêts aléatoires, puis optimisation du modèle par tuning des hyperparamètres._

→ **Jeu de données** sur la détection de maladies du foie chez des patients :
- préparation des données :
   * nettoyage
   * traitement des valeurs aberrantes
   * analyse des corrélations 
   * séparation train/test
- application du bagging :
   * instanciation et entraînement du modèle
   * calcul des métriques classiques
- application des forêts aléatoires :
   * instanciation et entraînement du modèle
   * calcul des métriques
- comparaison entre métriques des deux modèles
- visualisation de l'influence des variables explicatives
- optimisation des hyperparamètres
- analyse des résultats et sélection du meilleur modèle.

→ **Outils** : `Python` (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`), `Jupyter Notebook`.


## Projet 5 - Techniques avancées de visualisation de données

→ **Jeu de données** sur les soutenances de thèses en France :
- analyse et traitement des variables qualitatives et quantitatives
- amélioration esthétique des graphiques :
    * transparence
    * polices
    * marges
    * inclinaison des labels
    * position des légendes
    * palettes de couleurs
- graphiques interactifs avancés :
    * avec sliders
    * avec selectors

→ **Outils** : `Python` (`numpy`, `pandas`, `seaborn`, `matplotlib`, `plotly`), `Jupyter Notebook`, `GitHub`.


## Projet 6 - Analyse statistique avancée

_Analyse statistique approfondie utilisant des méthodes avancées pour explorer les relations, tester des hypothèses et modéliser un jeu de données complexe, dans le but de comprendre et d’expliquer les comportements d’engagement des apprenants dans le MOOC Effectuation._

→ **Jeu de données** sur le MOOC _Effectuation_ :
- analyse et traitement des variables :
   * fusion multi-sources
   * recodage
   * création de variables composites
   * harmonisation des formats
   * gestion des données manquantes
- Description des distributions :
   * moyennes
   * écarts-types
   * coefficients de variation
- Tests d’indépendance et comparaison de groupes :
  * test du chi² (calcul du V de Cramer)
  * tests non paramétriques (Mann-Whitney, Kruskal-Wallis)
  * ANOVA
- analyse de corrélation (corrélation de Spearman)
- modélisations statistiques
  * régressions (linéaire, logistique, de Poisson)
  * modèle de Cox (analyse de survie)
- Analyses des hazard ratios
- Visualisations synthétiques et détaillées illustrant :
  * distributions
  * comparaisons de groupes
  * évolutions temporelles
- Interprétation statistique :
  * p-values
  * intervalles de confiance
 
→ **Rapport d’analyse** au format IMRaD avec interprétations détaillées.

→ **Outils** : `Python` (`numpy`, `pandas`, `matplotlib`,`seaborn`, `scipy`, `statsmodels`, `missingno`, `math`, `lifelines`), `Jupyter Notebook`, `GitHub`.


## Projet 7 - Business Intelligence

_Analyse visant à étudier le lien entre le genre et diverses variables RH, afin d’optimiser les stratégies de gestion des ressources humaines._

→ **Jeu de données** RH :  
- extraction, transformation et nettoyage des données avec **Power Query**  
- sélection et transformation des variables pertinentes  
- création de visualisations dynamiques :  
  * cartes géographiques interactives  
  * tableaux de bord avec métriques clés (KPI)  
  * filtres interactifs pour faciliter l’exploration des données  
  * exploration des influenceurs clés (Key Influencers) pour identifier les facteurs impactant les indicateurs RH
 
→ **Tableau de bord** interactif.

→ **Outils** : Power BI (Power Query, modélisation, DAX, visualisations interactives, filtres)  


