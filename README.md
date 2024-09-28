<div style="bottom:50em; width: 35%; height: 35%; overflow: hidden"><img align="right" style="bottom:50em; width: 35%; height: 35%" src="https://github.com/user-attachments/assets/7ca6ec08-77e4-4782-920b-d548f9455786"></div>

# pybiv
Work with sums of bivariate functions in Python.
This packages provides software to approximate by, optimize and manipulate sums of bivariate functions.

#### Installation

`erdospy` can be installed directly from source with:
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
u = np.zeros((k,)*3)
for x in itertools.product(*list(map(range, (k,)*3))):
    u[x] = np.prod(x)/k**3

# create a plot of the residual of its best sum-of-bivariate-approximant
aprxmnt = approx(lambda x: np.prod(x)/k**3, (k,)*3)
ax.scatter3D(X, Y, Z, c=aprxmnt[2], alpha=0.5, marker='.')
plt.axis('off')
plt.grid(b=None)
plt.show()
```
