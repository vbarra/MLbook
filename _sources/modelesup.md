<div class="admonition note" name="html-admonition" style="background: lightblue; padding: 10px">

</div>


# Modèle du processus d'apprentissage supervisé
Un modèle classique de description du processus d'apprentissage supervisé est composé de trois composantes {cite}`Vapnik91`:
- Un environnement, qui fournit des vecteurs $\mathbf x\in X$ avec une probabilité fixe mais inconnue $P_X$ ;
- Un superviseur qui fournit pour chaque vecteur $\mathbf x$ reçu de l'environnement une réponse désirée $y\in Y$, selon une probabilité $P(\mathbf x\mid y)$ fixe mais inconnue. La réponse $y$ et $\mathbf x$ sont liés par une relation $y=f(\mathbf x,\epsilon)$, $\epsilon$ étant un bruit permettant au superviseur d'être "bruité" ;
- Un algorithme d'apprentissage qui implémente une classe de fonctions $\mathcal{F}$, définies par un paramètre vectoriel $\theta$, reliant l'espace des vecteurs $\mathbf x$ à l'espace des réponses $Y$ : $\mathcal{F} = \{F(\mathbf x,\theta),\theta\in\Theta\}$

Le problème de l'apprentissage supervisé consiste alors à choisir dans $\mathcal{F}$ une fonction qui approche la réponse $y$ pour tout $\mathbf x$ d'une manière optimale, au sens statistique du terme. La recherche de cet optimum est basée sur un ensemble de $n$ exemples i.i.d., dit ensemble  d'apprentissage $Z=\left \{(\mathbf x_i,y_i),1\leq i\leq n,\mathbf x_i\in X,y_i\in Y\right \}$. Chaque exemple $(\mathbf x_i,y_i)$ est tiré par l'algorithme d'apprentissage depuis $Z$ avec une probabilité jointe fixe mais inconnue $P_{X,Y}$. 

Trouver un "bon" candidat dans $\mathcal{F}$ qui approche $f$ repose sur le fait que $Z$ contient "suffisamment" d'information pour permettre d'une part d'apprendre correctement $Z$ (facile), mais aussi d'être capable de généraliser de manière cohérente sur $X\times Y$. La quantification de cette information a été apportée par les travaux de Vapnik et Chervonenkis {cite}`Vapnik71`.

Soit $L\left (y,F(\mathbf x,\theta)\right )$ une fonction de perte, qui mesure l'écart entre la réponse $y$ fournie par le superviseur et la réponse calculée par l'algorithme d'apprentissage. L'espérance de $L$ définit le risque fonctionnel

$$R(\theta) = \int L\left (y,F(\mathbf x,\theta)\right )dP_{X,Y}$$

que l'algorithme d'apprentissage doit donc minimiser sur la classe des fonctions $\mathcal{F}$. 

Cette minimisation est difficile, la probabilité $P_{X,Y}$ étant inconnue. La seule connaissance sur les couples $(\mathbf x,y)$ est contenue dans $Z$, et on remplace le problème de minimisation précédent par la minimisation du risque empirique :

$$R_{emp}(\theta) = \frac{1}{n}\displaystyle\sum_{i=1}^n L\left (y_i,F(\mathbf x_i,\theta)\right )$$

qui ne nécessite pas la connaissance de $P_{X,Y}$.

# Minimisation du risque empirique
Soit $\hat{\theta}$ (respectivement $F(\mathbf x\,\hat{\theta})$) le vecteur (resp. la fonction) optimal(e) pour le problème de minimisation du risque empirique. Pour une valeur $\theta^*\in\Theta$, le risque $R(\theta^*)$ est l'espérance d'une certaine variable aléatoire définie par $M_{\theta^*} = L(y,F(\mathbf x,\theta^*))$. Le risque empirique $R_{emp}(\theta^*)$, quant à lui, est la moyenne arithmétique de $M_{\theta^*}$. D'après la loi des grands nombres, si $n\rightarrow\infty$, la moyenne de $M_{\theta^*}$ tend vers son espérance, et donc vers $R(\theta^*)$, ce qui justifie d'utiliser le risque empirique en lieu et place de $R(\theta)$.

Il n'y a cependant aucune raison a priori pour que $\hat{\theta}$ minimise également $R(\theta)$.

Nous allons montrer que si $R_{emp}(\theta)$ approche uniformément $R(\theta)$ avec une précision $\epsilon$, alors le minimum du risque empirique s'écarte du minimum de $R(\theta)$ d'au plus $2\epsilon$.

Supposons 

$$(\forall \epsilon>0)\quad (\forall \theta\in\Theta)\quad \displaystyle\lim\limits_{n\rightarrow\infty}P\left (\displaystyle\sup_{\theta}\mid R(\theta)-R_{emp}(\theta)\mid>\epsilon\right ) =0$$

De manière équivalente, puisque pour tout $\epsilon>0$ on a $P\left (\displaystyle\sup_{\theta}\mid R(\theta)-R_{emp}(\theta)\mid>\epsilon\right ) <\alpha$ pour $\alpha>0$, alors 

$$P\left (\mid R(\hat{\theta})-R(\theta_0)\mid  >2\epsilon\right )<\alpha$$ 

$\theta_0$  minimisant $R(\theta)$. Ainsi, si $P\left (\displaystyle\sup_{\theta}\mid R(\theta)-R_{emp}(\theta)\mid>\epsilon\right ) <\alpha$ est vraie, alors avec probabilité au moins 1-$\alpha$ la fonction $F(\mathbf x,\hat{\theta})$ donnera un risque $R(\hat\theta)$ qui s'écartera du minimum $R(\theta_0)$  d'au plus $2\epsilon$. 

En effet on a 

- avec probabilité 1-$\alpha$ $\mid R(\hat\theta)-R_{emp}(\hat\theta)\mid<\epsilon$ et  $\mid R(\theta_0)-R_{emp}(\theta_0)\mid<\epsilon$
- $\hat{\theta}$ et $\theta_0$ étant les optimum de $R_{emp}$ et $R$, $R_{emp}(\hat{\theta})<R_{emp}(\theta_0)$

Ces trois équations permettent alors d'écrire 

$$\mid R(\hat\theta)-R(\theta_0)\mid<2\epsilon$$

La minimisation du risque empirique consiste donc à :

1. Calculer le risque empirique sur $Z$ et la valeur $\hat\theta$ de son paramètre réalisant le minimum de ce risque
2. Affirmer que $R(\hat\theta)$ converge en probabilité vers le risque minimum sur tout $X\times Y$, lorsque la taille de l'ensemble d'apprentissage tend vers l'infini, et ce en supposant que $R_{emp}(\theta$ converge uniformément vers $R(\theta)$
\end{enumerate}


# VC dimension
## Définition
La VC dimension (ou dimension de Vapnik-Chervonenkis) est une mesure du pouvoir d'expression d'une famille de fonctions de classification réalisées par un algorithme d'apprentissage.

Considérons un problème de classification binaire, i.e. le superviseur assigne chaque $\mathbf x$ à une des deux classes, notées 0 et 1. Une fonction de classification binaire est appelée une dichotomie. Un algorithme d'apprentissage construit alors un ensemble de dichotomies $\mathcal{F} =\left  \{F:X\times \Theta\rightarrow \{0,1\}\right \}$.

On note $\mathcal{L} = \left \{\mathbf x_i\in X,1\leq i\leq n \right \}$ un ensemble de $n$ points de $X$ . Une dichotomie réalisée par l'algorithme d'apprentissage partitionne $\mathcal{L}=\mathcal{L}_0\cup \mathcal{L}_1$  de sorte que $F(\mathbf x,\theta)=k$ si $\mathbf x\in \mathcal{L}_k, k\in\{0,1\}$.

On note $\Delta_\mathcal{F}(\mathcal{L})$ le nombre de dichotomies distinctes réalisées par l'algorithme et on définit la fonction de croissance $\Delta_\mathcal{F}:\mathbb{N}\rightarrow\mathbb{N}$ par 

$$\Delta_\mathcal{F}(l)=\displaystyle\max_{\mathcal{L},\mid \mathcal{L}\mid=l}\Delta_\mathcal{F}(\mathcal{L})$$

On dit que $\mathcal{L}$ est pulvérisé par $\mathcal{F}$ si $\Delta_\mathcal{F}(\mathcal{L})=2^{\mid \mathcal{L}\mid}$, i.e. si toutes les dichotomies possibles sur $\mathcal{L}$ peuvent être décrites par des fonctions de $\mathcal{F}$. 


````{prf:definition} VC-dimension.
La dimension de Vapnik-Chervonenkis (VC dim) d'un ensemble de dichotomies $\mathcal{F}$ est la cardinalité du plus grand ensemble $\mathcal{L}$ pulvérisé par $\mathcal{F}$.
````

En d'autres termes, la VC-dim de $\mathcal{F}$ est le plus grand $n$ tel que $ \Delta_\mathcal{F}(n)=2^n$, ou encore le nombre maximum d'exemples d'apprentissage qui peuvent être appris par l'algorithme sans erreur pour tous les étiquetages binaires possibles des fonctions de classification $\mathcal{F}$.

La VC-dim joue un rôle important dans la théorie de l'apprentissage statistique. Ainsi, par exemple, le nombre d'exemples nécessaires pour apprendre de manière fiable une classe de fonctions est proportionnel à la VC-dim de cette classe. 

Cette dimension est difficile à évaluer analytiquement. Cependant, quelques résultats existent et par exemple : 
1. si l'algorithme d'apprentissage est un réseau de neurones, dont les cellules ont des fonctions d'activation de Heaviside, alors la VC-dim associée est en $O(W log(W))$, où $W$ est le nombre de paramètres libres du réseau
2. Si l'algorithme d'apprentissage est un perceptron multicouches, dont les cellules ont des fonctions d'activation sigmoïde, alors la VC-dim associée est en $O(W^2)$, où $W$ est le nombre de paramètres libres du réseau

## Une application : minimisation du risque structurel
L'erreur d'entraînement $E_t(\theta)$ d'un algorithme d'apprentissage est reliée à la fréquence des erreurs obtenues par cet algorithme sur $Z$. On demande à un tel algorithme non seulement d'avoir une faible erreur d'entraînement, mais aussi d'être capable de donner des valeurs justes, sur des données non vues lors de la phase d'entraînement. On parle de bonne capacité de généralisation.\\
L'erreur en généralisation $E_g(\theta)$ mesure les erreurs effectuées par l'algorithme sur des exemples qu'il n'a jamais vu, appelés exemples test. On suppose que des exemples sont issus de la même population que les données d'entraînement. 

Soit $h$ la dimension de Vapnik-Chervonenkis d'une famille de classifieurs $\mathcal{F}$. On peut montrer {cite}`Vapnik91`  qu'avec probabilité $1-\alpha$, pour un nombre d'exemples $n>h$ que pour toutes les fonctions de $\mathcal{F}$, 

$$E_g(\theta) = E_t(\theta) + \epsilon_1\left (n,h,\alpha,E_t(\theta)\right )$$

avec

$$ \epsilon_1\left (n,h,\alpha,E_t(\theta)\right ) = 2\epsilon_0^2(n,h,\alpha)\left (1+\sqrt{1+\frac{E_t(\theta)}{\epsilon_0^2(n,h,\alpha)}}\right )$$

et où 

$$\epsilon_0(n,h,\alpha) = \sqrt{\frac{h}{n}\left ( log\left ( \frac{2n}{h}\right ) +1 \right ) -\frac{1}{n}log\alpha}$$

est l'intervalle de confiance. Pour $n$ fixé, $E_t(\theta)$ décroît lorsque $h$ augmente, alors que l'intervalle de confiance croît. Ainsi, le risque garanti et l'erreur en généralisation passent par un minimum. Avant cet optimum, le problème d'apprentissage est surdéterminé ($h$ est trop petit par rapport au niveau d'information contenu dans $Z$). Au-delà, il est sous-déterminé (l'algorithme est trop "complexe") ({numref}`vcdim-ref`).

```{figure} ./images/vcdim.png
:name: vcdim-ref
Relation entre les erreurs et $h$. L'erreur d'estimation donne une mesure de la performance perdue par la fonction d'approximation en utilisant un ensemble d'apprentissage de taille $n$. L'erreur d'approximation donne une mesure de performance en fonction de la complexité du modèle
```


Un des objectifs dans la résolution d'un problème d'apprentissage supervisé est donc d'atteindre la meilleure capacité de généralisation (minimiser $E_g(\theta)$). La minimisation du risque structurel propose une méthode inductive permettant d'atteindre cet objectif, en faisant de $h$ une variable de contrôle.

Soit pour $k\in\{1\ldots n\}$ $\mathcal{F}_k=\{F(\mathbf x,\theta),\theta\in\Theta_k\}$ un ensemble emboité de classes de fonctions tel que $\mathcal{F}_1\subset\mathcal{F}_2\ldots \mathcal{F}_n$. Les dimensions de Vapnik-Chervonenkis correspondantes vérifient $h_1\leq h_2\ldots h_n$. La minimisation du risque structurel consiste à : 

- Minimiser $E_t(\theta)$ pour chaque fonction
- Idenfifier la classe $\mathcal{F}^*$ dont le risque garanti est le plus faible. Une des fonctions dans $\mathcal{F}^*$ fournit le meilleur compromis entre $E_t(\theta)$ (qualité d'approximation de $Z$) et intervalle de confiance (complexité de la fonction d'approximation). 

En pratique, la variation de $h$, et donc la création des ensembles emboîtés, peut être réalisée dans des réseaux de neurones à nombre de neurones cachés croissant. 





```{bibliography}
```
