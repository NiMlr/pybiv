# pybiv

<div style="width: 43%; height: 43%; overflow: hidden"><img align="right" style="width: 43%; height: 43%" src="https://github.com/user-attachments/assets/7ca6ec08-77e4-4782-920b-d548f9455786"></div>

Work with sums of bivariate functions in Python.
This packages provides software to approximate by, optimize and manipulate sums of bivariate functions.

A sum of bivariate functions is $F: \Omega \to \mathbb{R}$, where
* $\Omega := \Omega_0 \times \dots \times \Omega_{n-1}$, where $\Omega_i := \\{0, \dots, K_i-1\\}$, $K_i \in \mathbb{N}$,
* $\mathcal{V} := \\{0, \dots n-1\\}$,
* $\mathcal{E} := \\{(i,j) \in \mathcal{V} \times \mathcal{V} \mid i < j \\}$,
* $F(x_0, \dots, x_{n-1}) = \sum_{(i,j) \in \mathcal{E}} f_{i, j}(x_i, x_j), x \in \Omega$.

The $f_{i,j}, (i,j) \in \mathcal{E}$ is typically known or can be found by approximation.
When working with this package $f_{i,j}, (i,j) \in \mathcal{E}$ is represented as a dictionary with
keys $\mathcal{E}$ and values that are 2d-arrays $f_{i,j}(x_i,x_j)_{x_i=0,\dots, k_i-1; x_j=0, \dots,k_j-1}$.

#### Installation

`pybiv` can be installed directly from source with:
```sh
pip install git+https://github.com/NiMlr/pybiv
```

Test your installation with:
```python
import pybiv
pybiv.test.test_all()
```

## Tutorial

### Contents
[1. Approximation](#approximation)  
[2. Optimization](#optimization)  
[3. Signal Reconstruction](#signal-reconstruction)  

#### Approximation

We create an approximation by sums of bivariates of a discrete function that is the normalized product of three numbers.
The function is passed as a callable (alternatively `numpy.ndarray` is also possible).
We plot the residual of the approximation that resembles the title image of this README.

```python
import matplotlib.pyplot as plt
import numpy as np
import itertools
from pybiv.approximate import approx

# generate a grid
k = 128
z = np.linspace(0, 1, k)
x = np.linspace(0, 1, k)
y = np.linspace(0, 1, k)
X, Y, Z = np.meshgrid(x, y, z)

# create a figure
fig = plt.figure()
ax = plt.axes(projection="3d")

# compute values of 3-d function
G = lambda x: np.prod(x)/k**3

# approximate
F = approx(G, (k,)*3)[2]

# create a plot of the residual of its best sum-of-bivariate-approximant
ax.scatter3D(X, Y, Z, c=F, alpha=0.5, marker='.')
plt.axis('off')
plt.grid(b=None)
plt.show()
```

#### Optimization

The available optimizers are the following.

| Name | Description |  
|-------|-----------|
| [`cd`](https://github.com/NiMlr/pybiv/blob/main/pybiv/optimize/cd.py) | Coordinate descent for directly minimizing sums of bivariates |
| [`lpdlp`](https://github.com/NiMlr/pybiv/blob/main/pybiv/optimize/lpdlp.py) | Linear programming for the Lagrangian dual of a relaxation of sums of bivariates paired with a heuristic to transform the solution |
| [`bcadtr`](https://github.com/NiMlr/pybiv/blob/main/pybiv/optimize/bcadtr.py) | Block coordinate ascent for the Lagrangian dual of a relaxation of sums of bivariates paired with a heuristic to transform the solution |
| [`trws`](https://github.com/NiMlr/pybiv/blob/main/pybiv/optimize/trws.py) | Sequential tree-reweighted message passing for the Lagrangian dual of a relaxation of sums of bivariates paired with a heuristic to transform the solution |
| [`trws_leg`](https://github.com/NiMlr/pybiv/blob/main/pybiv/optimize/trws.py) | Legacy sequential tree-reweighted message passing paired with a heuristic to transform the solution |

We generate a non-trivial mock problem.
Then we apply the `pybiv.optimize.trws` and `pybiv.optimize.bcadtr` to heuristically solve the NP-complete problem of minimizing the sum of bivariates $F:\Omega \to \mathbb{R}$.

```python
from pybiv.optimize import trws, bcadtr
import numpy as np

# number of arguments of F
n = 16
# number is discrete values each argument takes
np.random.seed(42)
K = np.random.randint(2, 10, n)
# indices of the summands
E = [(0, 8), (3, 4), (0, 6), (11, 14), (1, 15), (2, 7), (12, 14), (9, 10), (5, 14), (3, 13)]
# data struture representing all (compatible) bivariates
F = {edge: np.random.randn(K[edge[0]], K[edge[1]]) for edge in E}

# do 100 iterations and print the objective value of the approximate minimizer
print(trws(F, 100)[1])
# -17.376521549856204
print(bcadtr(F, 100)[1])
# -17.376521549856204
```

#### Signal Reconstruction

We model a signal reconstruction problem with constraints. In particular, the unknown signal is $\mathrm{sig}: S^1 \to \\{-2, -1, 0, 1, 2\\}$, i.e., it is periodic and takes discrete values. The goal is to recover the signal from a noisy version of itself using TV-regularizated $\ell^1$-minimization.

```python
import numpy as np
from pybiv.optimize import bcadtr
np.random.seed(0)

# create a signal with n values
n = 1000
A = 2.
freq = 3.
sig = np.round(A*np.sin(freq*2.*np.pi*np.arange(n)/n))

# impose noise
noise = 3.5
noisysig = sig + noise*(np.random.rand(n)-(1./2))


reg = 3.5
# encode domain
vals = np.unique(sig)
domain = np.arange((vals[0]-noise).astype("int"), (vals[-1]+noise).astype("int")+1)
m = len(domain)-1
# model regularization
a = np.arange(m,-1, -1) + np.arange(0,m+1)[:,None]
eye1 = np.abs(np.arange(-m,m+1))[a.astype("int")].astype("int")
# model circular graph
edges = [(i, i+1) for i in range(n-1)]
edges.append((n-1, 0))
# model cost functions as bivariates
f = {edge: np.abs(noisysig[edge[0]]-domain)[:,None] \
              +reg*eye1 for edge in edges}
f[(0, n-1)] = f[(n-1, 0)].T
del f[(n-1, 0)]
edges.pop(-1)
edges.append((0,n-1))
# coordinate priority (technical)
c = np.arange(n)

# compute the approximate minimum of the bivariate
res = bcadtr(f, 500, c=c)

# decode to receive the reconstruction of the signal
lpressol = np.array([res[0][i] for i in range(n)])
rcnstrctn = domain[lpressol]

# we identify >80% of the values correctly
print(np.linalg.norm(rcnstrctn-sig, ord=1))
# 174.0
```
![reconstruction for different regularizations](https://github.com/user-attachments/assets/f80c6afd-b412-4a86-8326-0ed8fb92860a)