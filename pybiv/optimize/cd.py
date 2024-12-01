import numpy as np
import scipy as sp
from copy import deepcopy
from ..tools import (orient, rename, get_c, get_nbrs)


def cd(f, B, c=None):
    """Try to minimize a sum of bivariate functions.

    Try to solve the NP-complete problem of minimizing
    a sum of discrete bivariate functions with
    by a coordinate descent.
    The sum is represented by its summands which are represented
    by 2d-arrays.

    Parameters
    ----------
    f : dict
        Dictionary with keys that are tuples of two integers and
        values that are 2-d `np.ndarray`s.
        Dimensions must be compatible, i.e.,
        the first and second dimension of the array correspondes to the
        first and second integer of the tuple, respectively, and
        the for each integer the dimension must be unique.

        This encodes discrete bivariate functions that can be summed, i.e.,
        a function that has a representation as a sum of functions,
        which each depend only on two of its arguments.
        

    B : tuple, optional
        Number of iterations of a descent on each coordinate
        sequentially in the order defined by `c`.
    
    c : dict
        Keys must be the integers in the keys of `f`.
        Maps each of these integers, resembling the dimensions
        of the problem, to a rank.
        An dimension with a smaller rank will a appear later in
        a coordinate-like minimization of the problem.

    Returns
    -------
    tuple
        An approximate minimizer of the sum of the bivariates.
    
    float
        The function value of the approximate minimizer.
    """
    # make sure vertices are index sequentially from zero--inplace
    rn1 = rename(f)
    rn1inv = {rn1[key]:key for key in rn1.keys()}
    # if not defined 
    if c is None:
        c, cinv = get_c(f)
    else:
        c = np.array([c[rn1[i]] for i in range(len(rn1))])
        cinv = np.argsort(c)

    # reindex them according to c
    rn2 = rename(f, {i: c[i] for i in range(len(c))})

    # make sure edges are oriented increasingly
    tp = orient(f)

    E = set(f.keys())
    n = max({v for edge in E for v in edge})+1

    # variable cardinality (Omega)
    K = {edge[i]:f[edge].shape[i] for edge in E for i in range(2)}

    # count number of neighbors and collect smaller and larger neighbors
    nbrs = get_nbrs(E, n)

    # (candidate) solutions, evaluations
    xstar = np.zeros(n, dtype=int)

    # preallocate
    m = {i: np.zeros(K[i]) for i in range(n)}

    def _eval(x):
        return np.sum([f[(i,j)][x[i], x[j]] for i,j in E])

    t = 0

    while t < B:
        # loop vertices in order defined by c (we reordered them)
        for i in range(n):
            m[i][:] = 0.
            for j in nbrs[i][0]:
                m[i] += f[(j,i)][xstar[j],:]
            for j in nbrs[i][1]:
                m[i] += f[(i,j)][:,xstar[j]]
            xstar[i] = np.argmin(m[i])

        t += 1

    fxstar = _eval(xstar)
    # transform to original vertex names and edge orders
    orient(f, tp)
    rename(f, rn2)
    rename(f, rn1)
    c = {rn1[i]: c[i] for i in range(len(rn1))}

    xstar = {rn2[i]: xstar[i] for i in range(len(rn2))}
    xstar = {rn1[key]: xstar[key] for key in xstar.keys()}

    return xstar, fxstar

