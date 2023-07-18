Les espaces de Hilbert à noyau reproduisant (EHNR) sont des espaces d'hypothèses ayant de bonnes propriétés vis à vis du problème d'apprentissage. La principale de ces propriétés est la propriété de reproduction, qui relie les normes dans l'espace de Hilbert à l'algèbre linéaire. 


# Espaces de Hilbert à noyau reproduisant

## Rappel sur les espaces de Hilbert
On suppose connues ici les notions de normes, d'espaces vectoriels normés et de produit scalaire.
````{prf:definition}
Soit $H$ un espace normé. Un sous-ensemble $M\subset H$ est dense dans $H$ si $H=\bar{M}$, où  $\bar{M}$ est la fermeture de $M$ dans $H$.
````

````{prf:theorem} Théorème de projection
Soit $M$ un sous-espace vectoriel fermé de l'espace vectoriel de $H$. Soit $\mathbf x \in H$.

Alors il existe un unique $\tilde{\mathbf x} \in M$ tel que $\displaystyle \left\|\mathbf x-\tilde{\mathbf x}\right\| = \inf_{\mathbf y \in H} \left\|\mathbf x-\mathbf y\right\|$. $\tilde{\mathbf x}$ est la projection orthogonale de $\mathbf x$ sur $M$ et est noté $P_{M}(\mathbf x)$.
````

Exemples:  
Soit $(\mathbf x_t)_{t\in T}$ une famille d'un espace vectoriel $H$ muni d'un produit scalaire $\left\langle .,.\right\rangle_H$.

-  L'espace engendré par  $(\mathbf x_t)_{t\in T}$ est l'ensemble des combinaisons linéaires finies de $\mathbf x_t, t \in T$ noté $Lin\left\{\mathbf x_t, t\in T \right\}$.
-  L'espace fermé engendré par  $(\mathbf x_t)_{t\in T}$ est le plus petit sous-espace fermé vectoriel de $H$ qui contient  $(\mathbf x_t)_{t\in T}$. On le note $\overline{Lin}\left\{\mathbf x_t, t\in T\right\}$.
-  Un ensemble $(\mathbf{e}_t)_{t\in T}$ d'éléments de $H$ est dit orthonormal si et seulement si $\forall (s,t) \in T^2, \left\langle \mathbf{e}_s , \mathbf{e}_t\right\rangle_H =\delta_{s,t}$.
-  Soit $\left\{\mathbf{e}_1,\dots, \mathbf{e}_k\right\}$ un système orthonormal de $H$. 

	Soit $M=\overline{Lin}\left\{\mathbf{e}_1,\dots,\mathbf{e}_k\right\}$. Alors $\forall \mathbf x \in H$ :
			* $P_{M}(\mathbf x)=\displaystyle\sum_{i=1}^k \left\langle \mathbf x,\mathbf{e}_i\right\rangle_H \mathbf{e}_i$
			* $\left\|P_{M}(\mathbf x)\right\|^2=\displaystyle\sum_{i=1}^k \left|\left\langle \mathbf x,\mathbf{e}_i\right\rangle_H \right|^2$
			* $\left\|\mathbf x-  \displaystyle\sum_{i=1}^k \left\langle \mathbf x,\mathbf{e}_i\right\rangle_H \mathbf{e}_i\right\| \leq  \left\|\mathbf x- \displaystyle\sum_{i=1}^k c_i \mathbf{e}_i\right\| \forall (c_1,\dots,c_k)\in \mathbb{R}^k $. 

On a égalité si $c_i=\left\langle \mathbf x,\mathbf{e}_i\right\rangle_H$.


````{prf:definition}
Un espace norm\'e $H$ est dit { séparable} si et seulement si il contient une suite dénombrable dense. Il est dit complet si et seulement si toute suite de Cauchy de $H$ converge dans $H$.
````

````{prf:definition}
Un espace vectoriel $A$, muni d'un produit scalaire, est un espace de Hilbert s'il est complet. Dans ce cas, s'il est séparable, il possède une base dénombrable.
````
Dans la suite, nous considérerons des espaces de Hilbert séparables.



## Ecriture simplifiée du noyau reproduisant

Soit $H$ un espace de Hilbert de fonctions : $f :E\rightarrow \mathbb R$.


````{prf:definition} Noyau reproduisant
$\begin{array}{l r c c l r} \text{Une fonction } & K : &E*E &\rightarrow        &\mathbb R  &\text{ est un noyau reproduisant de } H \text{ si et seulement si :}
                                               &     &(s,t) &\mapsto            & K(s,t) &
                                         \end{array}$ 
1. $\forall t \in E,\; K (\bullet,t) : E \rightarrow \mathbb R ,s \mapsto K(s,t)$ est un élément de $H$. 
  2. $\forall t \in E , \forall \varphi \in H, \left\langle \varphi,K(\bullet,t)\right\rangle_H = \left\langle \varphi(\bullet),K(\bullet,t)\right\rangle_H  =\varphi(t)$. 

  On reproduit $\varphi$ par produit scalaire.                                         
````


Exemple : orthonormlisation de Gram-Schmidt

Soit $H$ un espace de Hilbert de dimension $n$ finie avec une base $(\mathbf{f}_1,\dots,\mathbf{f}_n)$.
Le produit scalaire sur $H$ est alors défini  par les nombres $g_{i,j}=\left\langle \mathbf{f}_i,\mathbf{f}_j \right\rangle_H \text{ pour tout } i,j=1\dots,n$. 

La matrice G définit le produit scalaire  et s'appelle la matrice de base de Gram. 

Si $\mathbf{f}=\displaystyle\sum_{i=1}^n a_i \mathbf{f}_i$ et $\mathbf{g}=\displaystyle\sum_{i=1}^n b_i \mathbf{f}_i$, alors $\left\langle \mathbf{f},\mathbf{g}\right\rangle=\displaystyle\sum_{i=1}^n \sum_{j=1}^n a_i b_j g_{i,j}$.

Prenons une base orthonormale $(\mathbf{e}_1,\dots,\mathbf{e}_n)$ de $H$.

La fonction $K:E*E\rightarrow\mathbb R$ donnée par $K(x,y)=\displaystyle\sum_{i=1}^n \mathbf{e}_i(x) \mathbf{e}_j(y)$ définit alors un noyau reproduisant de $H$. 


```{Note}
Tout espace de Hilbert de dimension finie admet un noyau reproduisant.
```


 Nous introduisons les EHNR de deux manières, une première abstraite et une seconde plus intuitive.
## Forme linéaire et noyau reproduisant
````{prf:definition}
Soit $L :H\rightarrow \mathbb R$ une forme linéaire. $L$ est continue (ou bornée) s'il existe $M>0$ tel que 

$$(\forall f\in H)\quad  |L(f)|\leq M\|f\|$$

$\|.\|$ étant la la norme de l'espace de Hilbert $H$.
````

````{prf:theorem} Théorème de Riesz

 Soit $L : H\rightarrow \mathbb R$ une forme linéaire. $L$ bornée. Il existe $K\in H$ telle que
  
 $$(\forall f\in H)\quad  L(f) = \left\langle K,f \right\rangle$$
````


````{prf:definition}
Soit $t\in X$. On appelle fonctionnelle d'évaluation linéaire $L_t$ (ou forme linéaire) une fonction :

$$
\begin{array}{lll}
L_t :H&\rightarrow& \mathbb R\\
	f&\rightarrow &f(t)
\end{array}
$$

linéaire par rapport à $f$.
````

Le théorème de Riesz permet alors d'affirmer que tout espace de Hilbert ayant une fonctionnelle d'évaluation linéaire bornée possède un élément qui évalue tous ses vecteurs par simple produit scalaire.

````{prf:definition}
Pour $t\in X$, soit $L_t$ une fonctionnelle d'évaluation linéaire bornée, et $H$ un espace de Hilbert. $H$, muni de $L_t$ est un espace de Hilbert à noyau reproduisant (EHNR), noté $H_K$
````

La notation indicée $K$ se réfère à la définition d'un noyau reproduisant $K$ :

````{prf:definition}
Le noyau reproduisant est une fonction $K:E\times E\rightarrow \mathbb R$, symétrique, semi-définie positive, c'est-à-dire vérifiant pour tous réels $a_i$ et tous vecteurs $t_i,t_j\in E$ 

$$\displaystyle\sum_{i,j=1}^n a_ia_j K(t_i,t_j)\geq 0$$
````



La relation entre $K$ et $ H$ se traduit par $K(s,t)=\left\langle K_s,K_t\right\rangle$ et $K(t,.)=K_t$

Il existe une relation très étroite entre un espace de Hilbert à noyau reproduisant et son noyau reproduisant associé, formalisée par le théorème suivant :
````{prf:theorem} Théorème d'Aronszajn
1. Pour tout espace de Hilbert à noyau reproduisant, il existe un noyau reproduisant unique. 
2. Réciproquement, étant donnée une fonction $K:E\times E\rightarrow \mathbb{R}$ symétrique, semi-définie positive, il est possible de construire un espace de Hilbert à noyau reproduisant ayant $K$ pour noyau reproduisant.
\end{enumerate}
````

Nous présentons ici des éléments de preuve.  Si $H_K$ est un EHNR, il existe $K_t$, un représentant de l'évaluation de tout $t$. Définissons alors $K(s,t) = \left\langle K_s,K_t\right\rangle$. On a alors directement :
$
\begin{array}{ll}
 \left \| \displaystyle\sum_j a_jK_{t_j}\right \|^2&\geq &0\\
 \left \| \displaystyle\sum_j a_jK_{t_j}\right \|^2&= &\displaystyle\sum_{i,j}a_ia_j\left\langle K_{t_i},K_{t_j}\right\rangle\\
 \displaystyle\sum_{i,j}a_ia_jK(t_i,t_j)&= &\displaystyle\sum_{i,j}a_ia_j\left\langle K_{t_i},K_{t_j}\right\rangle\\
\end{array}
$

et $K$ est semi définie positive.

Réciproquement, soit un noyau reproduisant $K(.,.)$, on définit pour tout $t\in E K_t(.)=K(t,.)$ On montre alors que l'on peut simplement constuire un espace de Hilbert à noyau reprosuisant $H_K$ à partir de l'ensemble des fonctions formées par combinaison linéaire des fonctions $K_{t_i}$, muni du pruduit scalaire 

$$\left\langle  \displaystyle\sum_ia_iK_{t_i},\displaystyle\sum_ia_iK_{t_i}\right\rangle=\displaystyle\sum_{i,j}a_ia_j\left\langle K_{t_i},K_{t_j}\right\rangle=\displaystyle\sum_{i,j}a_ia_jK(t_i,t_j)$$

Puisque $K$ est semi définie positive, le produit scalaire est bien défini et on peut vérifier que pour tout $f\in H_K$, $\left\langle K_t,f\right\rangle=f(t)$.


## Le noyau intégral
Soit $K:E\times E\rightarrow \mathbb{R}$ une fonction (noyau) symétrique continue. On définit pour $H\in L^2$ de dimension finie l'opérateur $L_K : H\rightarrow \mathbb{R}$ par :

$$L_K(f) = \int_E K(\bullet,t)f(t)dt$$

$K$ s'appelle le noyau intégral. Il est semi-défini positif si et seulement si $L_K$ l'est. Donc si $K$ est  semi-défini positif, les valeurs propres de $L_K$ sont positives. Notons les $\lambda_1\cdots \lambda_k$, et $\phi_1\cdots \phi_k$ les vecteurs propres associés. On a en particulier $\left \langle \phi_i,\phi_j\right \rangle=\delta_{ij}$


````{prf:theorem} Théorème de Mercer
Étant donnés les éléments propres de l'équation intégrale $L_K$, définie par un noyau symétrique, défini positif $K$, on peut écrire

$$(\forall s,t\in E) K(s,t) = \displaystyle\sum_{j=1}^k\lambda_j \phi_j(s)\phi_j(t)$$		

la convergence étant en norme $L_2$ sur $E$
````


On peut alors définir le EHNR comme l'espace des fonctions combinaisons linéaires des vecteurs propres de l'équation intégrale :

$$H_K = \left \lbrace  f, f(s) = \displaystyle\sum_j c_j\phi_j(s), \|f\|_{ H_K}<\infty\right \rbrace$$

où $\|f\|_{H_K}$ est définie par 

$$\|f(s)\|^2_{H_K}=\left\langle \displaystyle\sum_j c_j\phi_j(s),\displaystyle\sum_j c_j\phi_j(s) \right\rangle_{H_K}=\displaystyle\sum_j\frac{c_j^2}{\lambda_j}$$

et où 

$$\left\langle f,g\right\rangle_{H_K} = \left\langle \displaystyle\sum_j c_j\phi_j(s),\displaystyle\sum_j d_j\phi_j(s) \right\rangle_{H_K}=\displaystyle\sum_j\frac{c_jd_j}{\lambda_j}$$	


## Connexion avec les noyaux
On rencontre souvent le concept d'espace de Hilbert à noyau reproduisant sous le vocable "astuce du noyau'" [Kernel trick](kernelTrick.md), en particulier lorsque l'on étudie les Machines à Vecteurs de Support, ou plus généralement les méthodes à noyau. Les points $\mathbf x\in E\subset\mathbb{R}^d$ sont projetés dans un espace de grande dimension par les éléments propres du noyau reproduisant (la dimension de l'espace est égale au nombre de valeurs propres non nulles de l'opérateur intégral) :

$$\mathbf x\mapsto\phi(\mathbf x) = (\sqrt{\lambda_1}\phi_1(\mathbf x)\cdots \sqrt{\lambda_k}\phi_k(\mathbf x))$$


```{bibliography}
```
