Les machines à vecteurs de support (SVM - Support vector Machines ou Séparateur à Vaste Marge) {cite}`Vapnik95` ont été définies de nombreuses manières, et sont appliquées dans de nombreux domaines depuis quelques années.



Soit $Z = \{ \mathbf x_i , y_i\}, \ 1,\leq i\leq n, \mathbf x_i \in {R}^d, y_i \in \{-1,1\}\}$ un ensemble d'apprentissage pour un problème de classification binaire.



# SVM linéaire
## Hyperplan séparateur
Une approche traditionnelle pour introduire les SVM est de partir du concept d'hyperplan séparateur des exemples positifs et négatifs de l'ensemble d'apprentissage. On définit alors la marge comme la distance du plus proche exemple à cet hyperplan, et on espère intuitivement que plus grande sera cette marge, meilleure sera la capacité de généralisation de ce séparateur linéaire.\\
Un hyperplan de $\mathbb{R}^d$ est défini par 

$$\mathbf w^T\mathbf x + b = 0$$


$\mathbf w$ étant le vecteur normal à l'hyperplan. La fonction 

$$\label{eq:f} 
f({\mathbf x}) = \textrm{sign}( {\mathbf w^T\mathbf x} + b )
$$

permet, si elle sépare les données d'apprentissage, de les classifier correctement.  Un tel hyperplan, représenté par ($\mathbf w,b)$ peut également être exprimé par $(\lambda \mathbf w,\lambda b), \lambda\in\mathbb{R}$. Il est donc nécessaire de définir l'hyperplan canonique comme étant celui éloigné des données d'une distance au moins égale à 1. En fait, on impose qu'un exemple au moins de chaque classe soit à distance égale à 1. On considère alors le couple $(\mathbf w,b)$ tel que :

$$\mathbf w^T\mathbf x_i + b \ge +1 \ \ \textrm{si} \ \ y_i = +1 \\
\mathbf w^T\mathbf x_i + b \le -1 \ \ \textrm{si} \ \ y_i = -1
$$

ou de manière plus compacte

$$\forall i\quad y_i (\mathbf  w^T\mathbf x_i + b) \ge 1$$

Puisque l'on cherche à avoir la marge la plus grande possible, il est alors intéressant de calculer la distance, au sens de la norme euclidienne, d'un point $\mathbf x_i$ à cet hyperplan.  Cette distance est la longueur du vecteur reliant $\mathbf x_i$ à sa projection sur l'hyperplan, et est donnée par : 

$$d\Big( (\mathbf w,b) \ , \ \mathbf x_i \Big)
= \frac{ y_i (\mathbf w^T\mathbf x_i + b) }{ \parallel \mathbf w \parallel } \ge \frac{1}{ \parallel \mathbf w \parallel }
$$

Intuitivement, on veut trouver l'hyperplan qui maximise la cette distance, pour les $\mathbf x_i$ les plus proches. L'équation précédente permet d'affirmer que cela est réalisé en minimisant $\parallel \mathbf w \parallel$, sous les contraintes de bonne classification.

## Problème d'optimisation
Le problème s'écrit alors comme un problème de minimisation sous contraintes : 

$$\min_\mathbf w\in\mathbb{R}^d\; \parallel \mathbf w \parallel^2\\
\textrm{sous }y_i(\mathbf  w^T\mathbf  x_i) \geq 1,\quad 1\leq i\leq n\\
$$

En introduisant les multiplicateurs de Lagrange, le problème dual s'écrit :

$$
min  \ W(\alpha) = -\displaystyle\sum_{i=1}^n{\alpha_i} +
\frac{1}{2} \displaystyle\sum_{i=1}^{n}\displaystyle \sum_{j=1}^ny_iy_j\alpha_i\alpha_j(\mathbf x_i ^T \mathbf x_j)  $$
sous 

$$\displaystyle \sum_{i=1}^n y_i\alpha_i = 0 $$

$$(\forall1\leq i\leq n)\; 0 \le \alpha_i \le C$$

où $\mathbf {\alpha}$ est le vecteur des $n$ multiplicateurs de Lagrange à déterminer, et $C$ est une constante. En définissant la matrice $(H)_{ij} = y_iy_j(\mathbf x_i ^T \mathbf x_j)$ et \mathbf{1} le vecteur de $\mathbb{R}^n$ dont toutes les composantes sont égales à 1, le problème se réécrit comme un problème de programmation quadratique (QP) : 

$$
min \label{eq:qp1}  W(\alpha) = {-\alpha}^T \mathbf{1} + \frac{1}{2}\alpha^T H \alpha
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ & \\
\textrm{sous } \label{eq:qp2}  \alpha^Ty = 0
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ & \\
\label{eq:qp3}  {0} \le {\alpha} \le C\mathbf{1}
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ &
$$

pour lequel de nombreuses méthodes de résolution ont été développées.

En dérivant l'équation précédente, il est possible de montrer que l'hyperplan optimal (canonique) peut être écrit comme 

$$\label{eq:w} \mathbf w = \displaystyle\sum_{i=1}^n \alpha_i y_i \mathbf x_i
$$
et $\mathbf w$ est donc juste une combinaison linéaire des exemples d'apprentissage.

On peut également montrer que 

$$(\forall 1\leq i\leq n)\quad \alpha_i(y_i(\mathbf w^T \mathbf x_i + b) - 1) = 0 
$$

ce qui exprime que lorsque $y_i(\mathbf w ^T \mathbf x_i + b) > 1$, alors  $\alpha_i = 0$ : seuls les points d'apprentissage les plus proches de l'hyperplan (tels que $\alpha_i > 0$)  contribuent au calcul de ce dernier, et on les appelle les vecteurs de support.\\

En supposant avoir résolu le problème QP, et donc en disposant du $\mathbf {\alpha}$ qui permet de calculer le vecteur $w$ optimal, il reste à déterminer le biais $b$. Pour cela, en prenant un exemple positif  $\mathbf x^+$ et un exemple négatif  $\mathbf x^-$ quelconques, pour lesquels 

$$(\mathbf w ^T \mathbf x^+ + b) = +1 \\
(\mathbf w ^T \mathbf x^- + b) = -1
$$

on a 

$$b = - \frac{1}{2} ( \mathbf w ^T \mathbf x^+ + \mathbf w ^T \mathbf x^- )
$$

L'hyperplan ainsi défini a besoin de très peu de vecteurs de support (méthode éparse) ({numref}`svmlin-ref`). 


```{figure} ./images/svmLin.png
:name: svmlin-ref
Deux jeux de points à respectivement 50 et 500 points par classe, tirées selon les mêmes lois. Dans les deux cas, l'hyperplan est défini par un très faible nombre de vecteurs support (en vert)
```



En Scikit-learn 
```
from sklearn.svm import LinearSVC
X,y = ...
svm_clf = LinearSVC(random_state=0, tol=1e-050)
svm_clf.fit(X, y);
```

## Données non linéairement séparables

Il reste à préciser le rôle de la contrainte ${0} \le {\alpha} \le C\mathbf{1}$.
Lorsque $C\rightarrow\infty$, l'hyperplan optimal est celui qui sépare totalement les données d'apprentissage (si tant est qu'il existe). Pour des valeurs de $C$ "raisonnables", des erreurs de classification peuvent être acceptées par le classifieur (soft margin). Pour cela on introduit des variables d'écart $\xi_i$ :

$$\forall 1\leq i\leq n\quad y_i(\mathbf w ^T \mathbf x_i + b) > 1-\xi_i$$

Les vecteurs de support vérifient l'égalité, et les anciennes contraintes peuvent être violées de deux manières :
	- $(\mathbf x_i,y_i)$ est à distance inférieure à la marge, mais du bon côté de l'hyperplan
	- $(\mathbf x_i,y_i)$ est du mauvais côté de l'hyperplan

L'objectif est alors de minimiser  la moyenne des erreurs de classification $\displaystyle\sum_{i=1}^n \mathbf{1}_{\xi_i>0}$. Ce problème étant NP-complet (fonction non continue et dérivable), on lui préfère le problème suivant 

$$	Min \frac{1}{2}\mathbf w^T\mathbf w + C\displaystyle\sum_{i}^n \xi_i\\
	sous\;  y_i\left ( \mathbf w^T\mathbf x_i+b\right )= 1-\xi_i
$$

$C$ représente alors un compromis entre la marge possible entre les exemples et le nombre d'erreurs admissibles.
Nous illustrons dans la suite deux situations influencées par $C$ : 

- La {numref}`soft1-ref` présente une première illustration du rôle de $C$ : dans le cas de données linéairement séparables, un $C$ faible autorisera des  vecteurs à rentrer dans la marge (vert). Plus $C$ devient grand, plus le nombre de vecteurs support diminue, pour ne laisser aucun vecteur à distance inférieure à la marge de l'hyperplan optimal

```{figure} ./images/soft1.png
:name: soft1-ref
Données linéairement séparables
```
- La {numref}`soft2-ref` présente un ensemble de données non linéairement séparables. La valeur de $C$ contrôle le nombre d'erreurs de classification dans le résultat final.

```{figure} ./images/soft2.png
:name: soft2-ref
Données non linéairement séparables
```

En Scikit-learn 
```
from sklearn.svm import SVC
X,y = ...
svm_clf = SVC(kernel="linear", C=1E10)
svm_clf.fit(X, y);
```

## Cas multiclasses
Deux stratégies sont possibles dans le cas multiclasse : 

````{tabbed} Un contre tous
Transformer le problème à $k$ classes en $k$ classifieurs binaires, la classe de l'exemple est donnée par le classifieur qui répond le mieux

![](images/ova1.png)
````

````{tabbed} Un contre un
Transformer le problème en $\frac{k(k-1)}{2}$ classifieurs binaires, chaque classe étant comparée aux autres. La classe de l'exemple est donnée par le vote majoritaire ou par un graphe acyclique de décision

![](images/ova2.png)

````


# SVM non linéaire
Pour utiliser les SVM dans un contexte non linéaire, on profite de l'[astuce du noyau](kernelTrick.md) puisque le modèle s'écrit avec un produit scalaire canonique.

```{margin} Exemple de regression
![](images/ridgeregression.png)
```


# Utilisation en régression
Il est également possible, en changeant les fonctions de perte,  d'utiliser les SVM  en régression non paramétrique ([SVR](SVR.ipynb) : Support Vector Regression)
, i.e. approcher une fonction de $\mathbb{R}^d$ dans $\mathbb{R}^p$ par les mêmes mécanismes d'optimisation. 



```{bibliography}
```
