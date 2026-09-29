# Sujets de projet, cours d'apprentissage automatique 2026-2027

L'objectif de ces projets est de proposer des études de régression et/ou de classification  sur des jeux de données ouverts, accessibles par API. Les API proposées exposent des historiques complets enregistrés point par point, permettant de constituer une base d'apprentissage, une base de test et une base de validation (pour l'optimisation des hyperparamètres). 

Vous établirerez un pipeline de traitement des données (récupération et préparation des données, extraction/sélection de variables, création de variables métier pertinentes, comparaison de modèles de régression et de classification à l'aide de mesures de performance, analyse de l'importance des variables d'entrée discriminantes, synthèse finale de l'étude).

**Livrables**
-  un notebook Jupyter avec le code, utilisant scikit-learn, et des commentaires *pertinents*. 
-  un modèle de référence basique (base classifier ou base regressor) avant de tester des modèles plus complexes.
-  un script de sauvegarde locale après le requêtage de l'API pour éviter de surcharger les serveurs et limiter le temps d'attente lors de la phase de modélisation.
- si l'IA est utilisée pour le codage, vous **devez** faire une analyse critique du code produit, en précisant le(s) IA utilisée(s) et en soulignant les points positifs/négatifs liés à la génération de votre code.



---

## Sujet 1 — Estimation de la Performance Énergétique de Logements 

* **Objectifs :** 
  * **en régression :** prédiction de la consommation d'énergie primaire ($kWh/m^2/an$).
  * **en classification :** classification DPE (Classes A à G).
* **API**  : [DPE Logements (ADEME / data.gouv.fr)](https://data.ademe.fr/data-fair/api/v1/datasets/dpe-v2-logements-existants/lines)
* **Documentation :** [Portail ADEME](https://data.ademe.fr/datasets/dpe03existant)
* **Points d'attention :**
  1. préparation des données : traitement des valeurs manquantes et filtrage des incohérences physiques (surfaces nulles, hauteurs aberrantes).
  2. Gestion des descripteurs : création de ratios métiers (ratio surface vitrée/surface habitable, isolation relative, rapport surface/volume,...).

---

## Sujet 2 — Prédiction de la Valeur Foncière Enrichie par l'Environnement Local

* **Objectif :** prédiction du prix net vendeur d'une transaction immobilière
* **APIs**  :
  * [API DVF (Etalab / Cerema)](https://dvf-api.etalab.gouv.fr) 
  * [API Overpass](https://overpass-api.de/api/interpreter)
* Documentations 
  * [API DVF](https://www.data.gouv.fr/dataservices/api-donnees-foncieres)
  * [Documentation Overpass](https://wiki.openstreetmap.org/wiki/Overpass_API)
* **Points d'attention :**
  1. Enrichissement spatial : récupération des transactions puis requêtage de l'API Overpass pour calculer la densité d'infrastructures à proximité (commerces, transports, écoles dans un rayon de 300m et 800m).
  2. Traitements des valeurs aberrantes : nettoyage du bruit des transactions réelles (ventes à 1€ symbolique, ventes groupées, montants aberrants).
  3. Encodage : encodage des données géographiques (latitude/longitude) et d'agencement.

---

## Sujet 3 — Analyse de la Sévérité des Accidents de la Circulation

* **Objectif :** classification ordinale ou multiclasse : indemne, blessé léger, hospitalisé, décédé
* **API :** [API Accidents de la Circulation](https://opendata.paris.fr/api/v2/catalog/datasets/accidentologie0/)
* Documentation : [Données BAAC](https://www.data.gouv.fr/fr/datasets/base-de-donnees-accidents-corporels-de-la-circulation)
* **Points d'attention :**
  1. Gestion du déséquilibre de classes : les accidents graves/mortels sont minoritaires. 
  2. Encodage et reclassification : vectorisation des variables qualitatives codées sous forme numérique (type de collision, tranche d'âge, catégorie de véhicule).
   
---

## Sujet 4 — Profilage de la Qualité Physico-Chimique de l'Eau des Rivières

* **Objectifs :** 
  * **en régression :** prédiction de la concentration en nitrates ou carbone organique dissous.
  * **en classification :** évaluation de la qualité globale (Bonne / Moyenne / Mauvaise).
* **API  :** [API Hub'Eau](https://hubeau.eaufrance.fr/api/v1/qualite_rivieres/analyse)
* **Documentation :** [Portail API Hub'Eau](https://hubeau.eaufrance.fr/page/api-qualite-cours-d-eau)
* **Point d'attention :** transformation de données depuis une structure de type "plusieurs paramètres physico-chimiques par station" vers une structure de matrice de caractéristiques (ligne = prélèvement, colonne = paramètre chimique).
---

## Sujet 5 — Diagnostic de Qualité et Profilage Nutritionnel des Produits 

* **Objectifs :** 
  * **en régression :** prédiction du score nutritionnel continu / Nutri-Score numérique (valeur dans [1,100]).
  * **en classification :** prédiction du groupe Nutri-Score (Classes A, B, C, D, E) ou le degré de transformation.
* **API :** [API Open Food Facts](https://world.openfoodfacts.org/api/v2/search)
* **Documentation :** [Documentation API Open Food Facts](https://openfoodfacts.github.io/api-documentation/)
* **Points d'attention :**
  1. Filtrage : interrogation de l'API sur des catégories spécifiques et extraction des attributs numériques uniquement.
  2. Imputation des données manquantes : traitement du bruit et des données manquantes fréquentes dans une base de données collaborative.
  3. Gestion des descripteurs :  calcul de ratios nutritionnels métiers (ratio sucre/fibres, densité calorique au 100g, ratio acides gras saturés/lipides totaux).

---

## Sujet 6 — Estimation du Risque Financier et Profilage des PME 


* **Objectifs :** 
  * **en régression :** prédiction du chiffre d'affaires estimé ou la taille du capital social.
  * **en classification :** prédiction de la tranche d'effectifs de l'entreprise (1-2 salariés, 3-5, 6-9, 10-19, etc.).
* **API  :** [API Recherche Entreprises](https://recherche-entreprises.api.gouv.fr/search)
* **Documentation :** [Portail API Recherche Entreprises](https://api.gouv.fr/les-api/api-recherche-entreprises)
* **Points d'attention :**
  1. Mise en pllace d'un jeu de données équilibré : requêtage itératif filtré par secteur d'activité (code NAF/APE) et par département pour constituer une matrice représentative du tissu économique.
  2. Transformation de Distribution : gérer les variables financières aux distributions très fortement étalées.
  3. Gestion des points aberrants : gérer le fait que certaines entreprises sont très grandes par rapport à d'autres
