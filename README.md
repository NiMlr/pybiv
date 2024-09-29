# pybiv

<div style="bottom:50px; width: 45%; height: 45%; overflow: hidden"><img align="right" style="width: 45%; height: 45%" src="https://github.com/user-attachments/assets/7ca6ec08-77e4-4782-920b-d548f9455786"></div>

Work with sums of bivariate functions in Python.
This packages provides software to approximate by, optimize and manipulate sums of bivariate functions.

A sum of bivariate functions is $f: \Omega \to \mathbb{R}$, where
* $\Omega = \Omega_0 \times \dots \times \Omega_{n-1}$, where $\Omega_i = \\{0, \dots, k_i-1\\}$, $k_i \in \mathbb{N}$,
* $\mathcal{V} := \\{0, \dots n-1\\}$,
* $\mathcal{E} := \\{(i,j) \in \mathcal{V} \times \mathcal{V} \mid i < j \\}$,
* $f(x_0, \dots, x_{n-1}) = \sum_{(i,j) \in \mathcal{E}} f_{i, j}(x_i, x_j), x \in \Omega$.

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
F = lambda x: np.prod(x)/k**3

# approximate
aprxmnt = approx(F, (k,)*3)

# create a plot of the residual of its best sum-of-bivariate-approximant
ax.scatter3D(X, Y, Z, c=aprxmnt[2], alpha=0.5, marker='.')
plt.axis('off')
plt.grid(b=None)
plt.show()
```
