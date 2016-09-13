# Implementations of label propagation like algorithms

This is a set of scikit-learn compatible implementations of label propagation (LP) like algorithms.
One can easily grid search and cross validate models using utils in scikit-learn.

## Implemented Algorithms

* Harmonic Function (HMN) [Zhu+, ICML03]
* Local and Global Consistency (LGC) [Zhou+, NIPS04]
* Partially Absorbing Random Walk (PARW) [Wu+, NIPS12]
* OMNI-Prop (OMNIProp) [Yamaguchi+, AAAI15]
* Confidence-Aware Modulated Label Propagation (CAMLP) [Yamaguchi+, SDM16]

## Example

```
In [1]: import numpy as np

In [2]: import networkx as nx

In [3]: from label_propagation import LGC

In [4]: from scipy.sparse import lil_matrix

In [5]: A = lil_matrix((4,4)) # adjacency matrix

In [6]: A[0,1]=1; A[1,0]=1

In [7]: A[1,2]=1; A[2,1]=1

In [8]: A[2,3]=1; A[3,2]=1

In [9]: A.todense() # simple undirected chain
Out[9]:
matrix([[ 0.,  1.,  0.,  0.],
        [ 1.,  0.,  1.,  0.],
        [ 0.,  1.,  0.,  1.],
        [ 0.,  0.,  1.,  0.]])

In [10]: x_train = np.array([1,2])

In [11]: y_train = np.array([0,1]) # node 1 -> label 0, node 2 -> label 1

In [12]: clf = LGC(graph=A, alpha=0.99)

In [13]: clf.fit(x_train,y_train) # scikit-learn compatible
Out[13]:
LGC(alpha=0.99,
  graph=<4x4 sparse matrix of type '<type 'numpy.float64'>'
  with 6 stored elements in LInked List format>,
  max_iter=30)

In [14]: x_test = np.array([0,3]) # to predict labels of node 0 and node 3

In [15]: clf.predict(x_test) # scikit-learn compatible
Out[15]: array([0, 1])
```
