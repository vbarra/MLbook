
# Unification

Tous les algorithmes précédents (et d'autres non abordés ici) sont des méthodes dont les solutions viennent de la résolution d'un problème aux éléments propres. Il s'agit de :

1. Calculer le noyau $\mathbf K\in\mathcal{M}_n(\mathbb R)$
2. Calculer ${\mathbf Y} = \Sigma  V^T$, $\mathbf V$ vecteurs propres de $\mathbf K$, $\mathbf \Sigma = diag(\lambda_i)$
3. Calculer une nouvelle valeur $\mathbf y=\Sigma^{-1}  \mathbf V^T  \mathbf K(\mathbf X,\mathbf x)$


Ainsi , en notant :


- $\mathbf 1=\left ( 1\cdots 1\right )^T\in\mathbb{R}^n$ 
- $\mathbf H=\mathbf I-\frac{1}{n}\mathbf 1\mathbf 1^T$
- $\mathbf A^{+}$ la pseudo inverse de $\mathbf A$


on a


| Méthode             | K                                                                                   |
|---------------------|-------------------------------------------------------------------------------------|
| ACP                 | $\mathbf X\mathbf X^T$                                                                           |
| MDS                 | $-\frac{1}{2}\mathbf H\mathbf D^{x}\mathbf H$, $\mathbf D^{x}$ matrice des distances entre les données              |
| ISOMAP              | $-\frac{1}{2}\mathbf H\mathbf D^{G}\mathbf H$,  $\mathbf D^{G}$ matrice des distances géodésiques entre les données |
| LLE                 | $\left ((\mathbf I- \mathbf W)(\mathbf I- \mathbf W)^T \right )^{+}$                                              |




