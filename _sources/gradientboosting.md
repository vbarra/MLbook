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



# Gradient Boosting

Nous avons souligné qu'AdaBoost pouvait être vu comme un algorithme d'optimisation de la fonction :

$$J(f) = \displaystyle\sum_{i=1}^n exp\left (-y_i f(({\mathbf x}_i) \right )$$

L'idée des méthodes de gradient boosting est de généraliser cette approche, en s'intéressant à d'autres fonctions de coût et à leurs gradients.


## Résidus et gradient

On apprend sur $Z$ une fonction (de régression, de classification) $h_1$, en utilisant un algorithme approprié. L'erreur commise par $h_1$, mesurée par la fonction de perte $\ell$, est :

$$E_{h_1} = \displaystyle\sum_{i=1}^n \ell \left (y_i,h_1(\mathbf{x}_i)\right )$$

La quantité $e_i=y_i-h_1(\mathbf{x}_i)$ est appelée le résidu de $h_1$ en $\mathbf{x}_i$. S'il est possible de trouver une fonction $\hat{h}$ telle que $\hat{h}(\mathbf{x}_i) = e_i$  pour tout $i\in[\![1, n]\!]$, alors la nouvelle fonction de régression $F=h_1+\hat{h}$ aura une erreur nulle sur tous les points de $\mathcal{S}$. 

La recherche de $\hat{h}$ étant difficile, on lui préfère la recherche d'une fonction $h_2$ telle que, pour tout $i\in[\![1,n]\!]$ $|h_2(\mathbf{x}_i) - e_i|<\epsilon$, $\epsilon>0$ petit. Dans ce cas, $F=h_1+h_2$ a une erreur $E_F$ plus petite que $E_{h_1}$.


Si par exemple la fonction de perte est définie par l'erreur quadratique : 

$$\ell\left (y,h_1(\mathbf{x})\right ) = \frac{1}{2}(y-h_1(\mathbf{x}))^2$$

alors le résidu s'écrit :

 $$e=y-h_1(\mathbf{x}) = -\frac{\partial}{\partial h_1(\mathbf{x})}\ell\left (y,h_1(\mathbf{x})\right )$$

et le résidu est alors l'opposé du gradient. 

Appliqués en $Z$, ces résidus définissent un nouvel ensemble $\tilde{Z} = \{\mathbf{x}_i,e_i\}_{1\leq i\leq n}$ sur lequel donc $h_2$ est appris.

## Algorithme de gradient boosting
L'idée précédente, qui consiste à écrire les résidus comme des gradients, peut être généralisée, et donne lieu à l'algorithme de gradient boosting présenté dans l'{prf:ref}`gradboosting-algorithm`.


```{prf:algorithm} 
:label: gradboosting-algorithm
**Entrée** : ${Z} = \{\mathbf{x}_i,y_i\}_{1\leq i\leq n}$, $T$, $\ell$

**Sortie** : $F$

1. Calculer une première hypothèse $h_1$ sur ${Z} $ 

2. Pour $t=2$ à $T$
	1. 	Calculer $(\forall i\in[\![1\cdots n]\!])\ e_i = -\frac{\partial}{\partial h_{t-1}(\mathbf{x}_i)}\ell(y_i,h_{t-1}(\mathbf{x}_i))$
	2.	Construire $\tilde{Z} = \{\mathbf{x}_i,e_i\}_{1\leq i\leq n}$
	3.	Apprendre $g$ sur $\tilde{Z}$
	4.	Calculer $\lambda_t = arg \displaystyle\min_{\lambda} \left (\displaystyle\sum_{i=1}^n \ell\left ( y_i,h_{t-1}(\mathbf{x}_i)+\lambda g(\mathbf{x}_i)\right ) \right )$
	5.	Définir $h_t = h_{t-1} + \lambda_t g$
3. $F = h_T$
```



La méthode agit de la même manière qu'une descente de gradient en ajustant l'hypothèse $h_t$ en fonction de l'opposé du gradient de la fonction de perte $\ell$. 

Nous avons introduit la méthode avec la fonction de perte quadratique, mais toute fonction de coût et les gradients associés peuvent être utilisés.


Parmi tous les algorithmes de gradient boosting, XGBoost (Extreme Gradient Boosting),  LightGBM, tous deux utilisant des arbres de régression comme hypothèses, ont démontré leur efficacité lors de nombreux défis. En 2018, CatBoost a permis d'adapter ce type d'approche à des données catégorielles.



