# Implementations of label propagation like algorithms

This is a set of scikit-learn compatible implementations of label propagation (LP) like algorithms.
One can easily grid search and cross validate models using utils in scikit-learn.

## Implemented Algorithms

* Harmonic Function (HMN) [Zhu+, ICML03]
* Local and Global Consistency (LGC) [Zhou+, NIPS04]
* Partially Absorbing Random Walk (PARW) [Wu+, NIPS12]
* OMNI-Prop (OMNIProp) [Yamaguchi+, AAAI15]
* Confidence-Aware Modulated Label Propagation (CAMLP) [Yamaguchi+, SDM16]

## Usage

### Example
```
python main.py hmn -g sample.edgelist -l sample.label -o sample.output
```

### Inputs
```
$ cat sample.edgelist  # [src node id] [dst node id]
0 1
1 2
2 3
$ cat sample.label  # [node id] [label id]
1 0
2 1
$ cat sample.modulation # KxK matrix (K: no. of labels)
0 1
1 0
```

### HMN

```
$ python main.py hmn -h
usage: main.py hmn [-h] -g GRAPHFILE -l LABELFILE [-o [OUTFILE]]

optional arguments:
  -h, --help            show this help message and exit
  -g GRAPHFILE, --graphfile GRAPHFILE
                        input graph file
  -l LABELFILE, --labelfile LABELFILE
                        input label file
  -o [OUTFILE], --outfile [OUTFILE]
                        output file (default=STDOUT)
```

### LGC

```
$ python main.py lgc -h
usage: main.py lgc [-h] -g GRAPHFILE -l LABELFILE [-o [OUTFILE]]
                   [--alpha [ALPHA]]

optional arguments:
  -h, --help            show this help message and exit
  -g GRAPHFILE, --graphfile GRAPHFILE
                        input graph file
  -l LABELFILE, --labelfile LABELFILE
                        input label file
  -o [OUTFILE], --outfile [OUTFILE]
                        output file (default=STDOUT)
  --alpha [ALPHA]       alpha (default=0.99)
```

### PARW

```
$ python main.py parw -h
usage: main.py parw [-h] -g GRAPHFILE -l LABELFILE [-o [OUTFILE]]
                    [--lamb [LAMB]]

optional arguments:
  -h, --help            show this help message and exit
  -g GRAPHFILE, --graphfile GRAPHFILE
                        input graph file
  -l LABELFILE, --labelfile LABELFILE
                        input label file
  -o [OUTFILE], --outfile [OUTFILE]
                        output file (default=STDOUT)
  --lamb [LAMB]         lambda (default=1.0)
```

### OMNIProp

```
$ python main.py omni -h
usage: main.py omni [-h] -g GRAPHFILE -l LABELFILE [-o [OUTFILE]]
                    [--lamb [LAMB]]

optional arguments:
  -h, --help            show this help message and exit
  -g GRAPHFILE, --graphfile GRAPHFILE
                        input graph file
  -l LABELFILE, --labelfile LABELFILE
                        input label file
  -o [OUTFILE], --outfile [OUTFILE]
                        output file (default=STDOUT)
  --lamb [LAMB]         lambda (default=1.0)
```

### CAMLP

```
$ python main.py camlp -h
usage: main.py camlp [-h] -g GRAPHFILE -l LABELFILE [-o [OUTFILE]]
                     [--beta [BETA]] [--modulationfile [MODULATIONFILE]]

optional arguments:
  -h, --help            show this help message and exit
  -g GRAPHFILE, --graphfile GRAPHFILE
                        input graph file
  -l LABELFILE, --labelfile LABELFILE
                        input label file
  -o [OUTFILE], --outfile [OUTFILE]
                        output file (default=STDOUT)
  --beta [BETA]         beta (default=0.1)
  --modulationfile [MODULATIONFILE]
                        modulation matrix file (default: use identity)
```

## Usage in code

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

