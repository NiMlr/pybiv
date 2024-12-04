import numpy as np
import scipy as sp
from copy import deepcopy
from ..tools import (orient, rename, get_c, get_nbrs)


def bcadtr(f, B, c=None, w="constant"):
    """Try to minimize a sum of bivariate functions.

    Try to solve the NP-complete problem of minimizing
    a sum of discrete bivariate functions with a
    legacy implementation of the BCADTR method.
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
        Number of iterations. Has roughly a complexity of
        O(|v|^2 k^2) flops per iteration, where |v| is the number of
        arguments/dimension of the problem and k is the maximum number values
        each argument can take.
    
    c : dict
        Keys must be the integers in the keys of `f`.
        Maps each of these integers, resembling the dimensions
        of the problem, to a rank.
        An dimension with a smaller rank will a appear later in
        a coordinate-like minimization of the problem.

    w : str
        Indicates whether distribution is constant or uniformly random.
        Must be either `"constant"` or `"random"`. Default is `"constant"`.

    Returns
    -------
    tuple
        An approximate minimizer of the sum of the bivariates.
    
    float
        The function value of the approximate minimizer.
    """
    # make sure vertices are index sequentially from zero--inplace
    rn1 = rename(f)
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
    
    mm = {(edge[i], edge[1-i]): np.zeros(K[edge[i]]) for edge in E for i in range(2)}
    rho = {(edge[i], edge[1-i]): np.zeros(K[edge[i]]) for edge in E for i in range(2)}
    m = {i: np.zeros(K[i]) for i in range(n)}
    r = {i: np.zeros(K[i]) for i in range(n)}

    
    # count number of neighbors and collect smaller and larger neighbors
    nbrs = get_nbrs(E, n)
    
    nn = [len(nbrs[v][0])+len(nbrs[v][1]) for v in range(n)]

    if w == "random":
        israndom = True
    else:
        israndom = False
    # one more to work for the random case
    w = {i: np.ones(nn[i]+1)/nn[i] for i in range(n)}

    def _sample(y, i):
        y[i][1:-1] = np.random.rand(nn[i]-1)
        y[i][0] = 0.
        y[i][-1] = 1.
        y[i].sort()
        y[i][:-1] = np.diff(y[i])

    # (candidate) solutions, optimization, evaluations
    x = np.zeros(n, dtype=int)
    xstar = np.zeros(n, dtype=int)
    def _opt():
        for i in range(n):
            r[i][:] = 0.
            for j in nbrs[i][1]:
                # r[i][:] += rho[(i,j)]
                r[i][:] += mm[(i,j)]
            for j in nbrs[i][0]:
                r[i][:] += f[(j,i)][x[j], :]
            x[i] = np.argmin(r[i])

    def _eval(x):
        return np.sum([f[(i,j)][x[i], x[j]] for i,j in E])
    fxstar = _eval(xstar)

    t = 0
    g = deepcopy(f)

    while t < B:
        # loop vertices in order defined by c /reversed c (we reordered them)
        for i in range(n):
            m[i][:] = 0.

            # loop "increasing"/"decreasing" edges wrt to c
            for j in nbrs[i][1]:
                # inplace
                g[(i,j)][:] = f[(i,j)]
                g[(i,j)] -= rho[(j,i)][None,:]
                np.min(g[(i,j)], axis=1, out=mm[(i,j)])
                m[i] += mm[(i,j)]

            for j in nbrs[i][0]:
                # inplace
                g[(j,i)][:] = f[(j,i)]
                g[(j,i)] -= rho[(j,i)][:,None]
                np.min(g[(j,i)], axis=0, out=mm[(i,j)])
                m[i] += mm[(i,j)]
            
            # distribute
            minm = np.min(m[i])
            if israndom:
                _sample(w, i)

            jc = 0
            for j in nbrs[i][1]:
                rho[(i,j)][:] = minm
                rho[(i,j)] -= m[i]
                rho[(i,j)] *= w[i][jc]
                rho[(i,j)] += mm[(i,j)]
                jc += 1
            for j in nbrs[i][0]:
                rho[(i,j)][:] = minm
                rho[(i,j)] -= m[i]
                rho[(i,j)] *= w[i][jc]
                rho[(i,j)] += mm[(i,j)]
                jc += 1
                
        
        _opt()
        fx = _eval(x)
        if fx < fxstar:
            xstar[:] = x.copy()
            fxstar = fx

        t += 1

    # transform to original vertex names and edge orders
    orient(f, tp)
    rename(f, rn2)
    rename(f, rn1)
    c = {rn1[i]: c[i] for i in range(len(rn1))}

    xstar = {rn2[i]: xstar[i] for i in range(len(rn2))}
    xstar = {rn1[key]: xstar[key] for key in xstar.keys()}

    return xstar, fxstar

