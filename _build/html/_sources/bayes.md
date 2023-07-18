---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Classifieur naïf de Bayes

Soit un ensemble d'apprentissage  $Z=\left \{(\mathbf x_i,y_i),1\leq i\leq n, \mathbf x_i\in X,y_i\in Y \right \}$.
Le classifieur naïf de Bayes est une méthode de classification supervisée qui repose sur une hypothèse
simplificatrice forte : les descripteurs (coordonnées sur $X$) sont deux à deux indépendants conditionnellement à la classe à prédire dans $Y$. 

Soit $\mathbf x$ un individu à classer. La règle bayésienne d'affectation optimale consiste à maximiser la probabilité a posteriori d'appartenance aux classes, i.e.

$(\mathbf x\textrm{ est dans la classe }k\in Y) \Leftrightarrow (k = arg \displaystyle\max_l P(y=l |\mathbf )) $

La décision repose donc sur une estimation de la probabilité conditionnelle $P(Y|\mathbf x)$, qui peut s'écrire par la règle de Bayes 

$(\forall l\in Y)\quad P(y=l |\mathbf x) = \frac{P(\mathbf x |y=l)P(y=l)}{P(\mathbf x)}$

La maximisation ne dépendant pas de $\mathbf x$, la maximisation s'écrit alors 

$(\mathbf x\textrm{ est dans la classe }k\in Y) \Leftrightarrow (k = arg \displaystyle\max_l P(\mathbf x |y=l)P(y=l)) $

Le problème de l'apprentissage d'une règle de classification est donc résolu  si l'on possède
une connaissance exacte pour chaque classe de la probabilité a priori
$P(y=l)$ et de la densité conditionnelle $P(x \mid y=l)$.  Cependant, on ne dispose en pratique que des données d'apprentissage $Z$.

## Estimation de la probabilité a priori des classes

Pour estimer les valeurs $P(y=l)$, les probabilités a priori des
classes, on peut procéder de plusieurs manières.

1. Soit on dispose de connaissances a priori sur les données,
extérieures à l'ensemble d'apprentissage $Z$. Dans ce cas, on  doit les utiliser.
Par exemple,  si on cherche à reconnaître les lettres manuscrites, on
doit se fier aux statistiques de leur apparition dans
les textes, même si cette proportion n'est pas respectée dans $Z$.
2. 
     Sinon, en l'absence d'information particulière, on peut les supposer égales entre elles
     et donc prendre l'estimateur : $ \widehat{P(y=l)} = \frac{1}{C}$, si $C$ est le nombre de classes
3. 
    On peut aussi (ce qui est fréquemment fait  implicitement)
    supposer l'échantillon d'apprentissage représentatif et les
    estimer par les fréquences d'apparition de chaque classe dans cet
    ensemble : $ \widehat{P(y=l)} = \frac{n_l}{n}$, où $n_l$ est le nombre d'éléments de la classe $l$ dans $Z$
4. 
    Il existe aussi un estimateur intermédiaire (formule de Laplace) :

    $ \widehat{P(y=l)} =\frac{n_l+M/C}{n+M}$
     où $M$ est un nombre arbitraire.  Cette formule est
    employée quand $n$ est petit, donc quand les estimations $n_l/n$ sont
    très imprécises.  $M$ représente une augmentation virtuelle du nombre
    d'exemples, pour lesquels on suppose les classes équiprobables.


Le second cas s'applique par exemple à la reconnaissance des chiffres
manuscrits sur les chèques ; en revanche, pour les codes postaux, la troisième
méthode est préférable si la base de données a été bien constituée
(la proportion de chiffres $0$ y est supérieure
à celle des autres).

Si le troisième cas semble plus naturel, il peut aussi être trompeur : dans
certains problèmes, les classes ne sont pas représentées de la même
manière dans l'ensemble d'apprentissage et dans les exemples qu'il faudra
classer.  Par exemple, un diagnostic médical peut s'apprendre à partir d'un
ensemble d'apprentissage comportant un nombre équilibré d'exemples et de
contre-exemples, alors que la maladie est rare. Il faudra alors corriger ce biais.


## Estimation des densités conditionnelles a priori

Il reste donc à estimer les densités $P({\mathbf x} \mid  
y=l)$, appelées aussi vraisemblances.
Dans un problème d'apprentissage de règle de classification, on dispose d'un
échantillon d'exemples supervisés : le problème se ramène donc à estimer chaque $P({\mathbf x} \mid  
y=l)$ à partir des échantillons d'apprentissage supervisés par la classe $l$.

Indépendamment pour chaque classe, on se trouve donc finalement à devoir
estimer une densité de probabilité à partir d'un nombre fini
d'observations.  C'est un problème tout à fait classique en
statistiques.  Il faut introduire des hypothèses supplémentaires pour le
résoudre (un biais, en termes d'apprentissage artificiel).  On a l'habitude
de distinguer :

- Les méthodes *paramétriques*, où l'on suppose que les
    $P({\mathbf x} \mid  y=l)$ possèdent une certaine forme
    analytique. En général, on fait l'hypothèse qu'elles sont des
    distributions gaussiennes.  Dans ce cas, le problème se ramène
    à estimer la moyenne et la covariance de chaque distribution ; la
    probabilité d'appartenance d'un point ${\mathbf x}$ à une classe
    se calcule alors directement à partir des coordonnées de
    ${\mathbf x}$.  
- Les méthodes *non paramétriques*, pour lesquelles on estime
    localement les densités  $P({\mathbf x} \mid  y=l)$ au point
    ${\mathbf x}$ en observant l'ensemble d'apprentissage autour  de ce point. Ces méthodes sont implémentées par la technique des fenêtres de Parzen ou l'algorithme des k-plus proches voisins
- Les méthodes *semi-paramétriques*, pour lesquelles on ne connaît pas non plus la forme analytique des
distributions de probabilités.  On suppose cependant que ces distributions appartiennent à des familles et que les "hyper-paramètres" qui les caractérisent à l'intérieur de cette famille peuvent être déterminés.


Dans le cas du classifieur naïf de Bayes, on suppose que les descripteurs sont deux à deux indépendants conditionnellement aux valeurs de la variable  de classe, et ainsi 

$P(\mathbf x |y=l)=\displaystyle\prod_{j=1}^d P\left (f_j=x^j| y=l\right )$

où $d$ est le nombre de descripteurs (la dimension de $X$), $x^j$ la $j^e$ coordonnée de $\mathbf x$ et $f_j$ est le $jê$ descripteur. Le nombre de paramètres à estimer est alors très réduit, et pour une variable quelconque pouvant prendre $Q$ valeurs, on utilise 

$(\forall q)\quad P\left (f_j=q| y=l\right ) = \frac{n_{lq}+m}{n_l+mQ}$

Le classifieur de Bayes opère généralement sur le logarithme des probabilités (ceci d'autant plus que $d$ est grand)  et la règle du classifieur naïf de Bayes est finalement : 

$(\mathbf x\textrm{ est dans la classe }k\in Y) \Leftrightarrow \left (k = arg \displaystyle\max_l \left [log(P(y=l) +\displaystyle\sum_{j=1}^d  log P\left (f_j=x^j| y=l\right )\right ]\right ) $

En reprenant la règle de Bayes 

$(\forall k\in Y)\quad P(y=l |\mathbf x) = \frac{P(\mathbf x |y=l)P(y=l)}{P(\mathbf x)}$

la quantité $P(y=l)$ définit la probabilité a priori (par exemple de la classe $l$). Une fois les données d'apprentissage observées, cette probabilité devient $P(y=l\mid \mathbf x)$, la probabilité a posteriori (par exemple de la  classe $l$). La règle bayésienne d'apprentissage désigne donc  la règle de plus forte probabilité a posteriori, que l'on appelle également règle du maximum a posteriori (MAP).

## Optimalité
La règle de classification bayésienne est optimale en moyenne : parmi toutes les classifications possibles, elle est celle qui minimise la probabilité d'erreur, connaissant la probabilité a priori des classes. L'erreur obtenue par la règle bayésienne est appelée erreur bayésienne de classification, et les algorithmes de classification supervisés visent souvent à l'approcher. 



## Quand utiliser un classifieur naïf de Bayes ?

Les classifieurs naïfs bayésiens ont bien entendu des limites, eu égard aux hypothèses de distribution des données qui sont faites. Cependant, ils présentent des avantages :

- ils sont rapides en apprentissage et prédiction
- ils permettent une interprétation probabiliste simple
- ils ont pas (ou peu) de paramètres, donc pas (ou peu) de réglages empiriques



Une stratégie possible en classification consiste par exemple à examiner le comportement d'un classifieur naïf bayésien. Si les résultats sont satisfaisants, pas la peine de chercher un modèle complexe.


Ces classifieurs donnent de bons résultats dans les cas suivants :
- lorsque l'hypothèse (gaussienne, multinomiale,...) correspond effectivement à la distribution des données (rare...)
- lorsque les données sont naturellement bien séparées
-  lorsque les données sont plongées dans des espaces de très grande dimension

## Exemple

On construit un classifieur naïf de Bayes qui va prédire le genre d'une personne à partir de son poids, de sa taille et de sa pointure de chaussure.

On construit tout d'abord $Z$ avec $n=8$
```{code-cell} ipython3
import numpy as np
import pandas as pd

data = pd.DataFrame()

data['Genre'] = ['femme','femme','femme','femme','homme','homme','homme','homme']

data['taille'] = [172,160, 180, 175, 173, 186, 177, 181]
data['poids'] = [80,55,80,70,76,100,80,83]
data['pointure'] = [41,38,43,41,41,45,44,45]
```

Le classifieur va permettre de prédire le genre d'une nouvelle personne $X$

```{code-cell} ipython3
X = pd.DataFrame()
X['taille'] = [171]
X['poids'] = [75]
X['pointure'] = [41]
```

On utilise le formalisme précédent en utilisant une hypothèse gaussienne pour la vraisemblance. 

${\displaystyle {P(\textrm{X est un homme} \mid X)}={\frac {P({\text{homme}})\,p({\text{taille}}\mid{\text{homme}})\,p({\text{poids}}\mid{\text{homme}})\,p({\text{pointure}}\mid{\text{homme}})}{P(X)}}}$

${\displaystyle {P(\textrm{X est une femme} \mid X)}={\frac {P({\text{femme}})\,p({\text{taille}}\mid{\text{femme}})\,p({\text{poids}}\mid{\text{femme}})\,p({\text{pointure}}\mid{\text{femme}})}{P(X)}}}$

On suppose que les valeurs des descripteurs d'une personne sont distribuées selon une loi normale et par exemple 

$
p(\text{poids}\mid\text{femme})=\frac{1}{\sqrt{2\pi.\text{variance des poids dans les données}}}\,e^{ -\frac{(\text{poids}-\text{poids moyen des femmes dans les données})^2}{2\text{variance des poids dans les données}} }
$

```{code-cell} ipython3
data_means = data.groupby('Genre').mean()
data_variance = data.groupby('Genre').var()

taille_H_mean = data_means['taille'][data_variance.index == 'homme'].values[0]
poids_H_mean = data_means['poids'][data_variance.index == 'homme'].values[0]
pointure_H_mean = data_means['pointure'][data_variance.index == 'homme'].values[0]

taille_H_variance = data_variance['taille'][data_variance.index == 'homme'].values[0]
poids_H_variance = data_variance['poids'][data_variance.index == 'homme'].values[0]
pointure_H_variance = data_variance['pointure'][data_variance.index == 'homme'].values[0]

taille_F_mean = data_means['taille'][data_variance.index == 'femme'].values[0]
poids_F_mean = data_means['poids'][data_variance.index == 'femme'].values[0]
pointure_F_mean = data_means['pointure'][data_variance.index == 'femme'].values[0]

taille_F_variance = data_variance['taille'][data_variance.index == 'femme'].values[0]
poids_F_variance = data_variance['poids'][data_variance.index == 'femme'].values[0]
pointure_F_variance = data_variance['pointure'][data_variance.index == 'femme'].values[0]
```

On calcule ensuite les vraisemblance


```{code-cell} ipython3
def vraisemblance(x, mean_y, variance_y):

   return 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
```

et on applique le classifieur naïf de Bayes pour estimer la classe de $X$

```{code-cell} ipython3

n_homme = data['Genre'][data['Genre'] == 'homme'].count()
n_femme = data['Genre'][data['Genre'] == 'femme'].count()
total = data['Genre'].count()

P_H= n_homme/total
P_F = n_homme/total


H = P_H * \
vraisemblance(X['taille'][0], taille_H_mean, taille_H_variance) * \
vraisemblance(X['poids'][0], poids_H_mean, poids_H_variance) * \
vraisemblance(X['pointure'][0], pointure_H_mean, pointure_H_variance)

F = P_F * \
vraisemblance(X['taille'][0], taille_F_mean, taille_F_variance) * \
vraisemblance(X['poids'][0], poids_F_mean, poids_F_variance) * \
vraisemblance(X['pointure'][0], pointure_F_mean, pointure_F_variance)

if F>H :
    print("la personne est une femme")
else:
    print("la personne est un homme")

```



