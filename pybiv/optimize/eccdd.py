import numpy as np
import scipy as sp
from copy import deepcopy
from ..tools import (orient, rename, get_c, get_nbrs)


def eccdd(f, B, eps, h=None):
    """Try to minimize a sum of bivariate functions.

    Try to solve the NP-complete problem of minimizing
    a sum of discrete bivariate functions with the TRW-S method.
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
    if h is None:
        c, cinv = get_c(f)
    else:
        # rename edges in h
        hh = deepcopy(h)
        for s in range(len(h)):
            for edge_t in h[s]:
                hh[s].remove(edge_t)
                hh[s].add((rn1inv[edge_t[0]],rn1inv[edge_t[1]]))
        h = deepcopy(h)

        # generate order on vertices from first appearances in h
        V = {v for key in f.keys() for v in key}
        c = np.zeros(len(V))
        i = 0
        for s in h:
            for edge_t in s:
                if edge_t[0] in V:
                    V.remove(edge_t[0])
                    c[edge_t[0]] = i
                    i += 1
        cinv = np.argsort(c)

    # make sure edges are oriented increasingly
    tp = orient(f)

    E = set(f.keys())
    n = max({v for edge in E for v in edge})+1


    # variable cardinality (Omega)
    K = {edge[i]:f[edge].shape[i] for edge in E for i in range(2)}
    
    rhominus = {(edge[i], edge[1-i]): np.zeros(K[edge[i]]) for edge in E for i in range(2)}
    rho = {(edge[i], edge[1-i]): np.zeros(K[edge[i]]) for edge in E for i in range(2)}
    r = {i: np.zeros(K[i]) for i in range(n)}
    d = {i: np.zeros(K[i]) for i in range(n)}

    
    # count number of neighbors and collect smaller and larger neighbors
    nbrs = get_nbrs(E, n)

    # build h from larger and smaller neighbor sets
    if h is None:
        h = []
        for i in range(n):
            h.append({(cinv[i],j) for j in nbrs[cinv[i]][0]})
        for i in range(n):
            h.append({(cinv[n-i-1],j) for j in nbrs[cinv[n-i-1]][1]})

    # define gamma
    gamma = np.zeros(n)
    for i in range(n):
        gamma[i] = (eps/(len(nbrs[i][0])+len(nbrs[i][1])))\
                    + eps*np.log((len(nbrs[i][0])+len(nbrs[i][1]))/eps)

    # (candidate) solutions, optimization, evaluations
    x = np.zeros(n, dtype=int)
    xstar = np.zeros(n, dtype=int)
    def _opt():
        for i in range(n):
            r[i][:] = 0.
            for j in nbrs[i][1]:
                r[i][:] += rho[(i,j)]
            for j in nbrs[i][0]:
                r[i][:] += f[(j,i)][x[j], :]
            x[i] = np.argmin(r[i])

    def _eval(x):
        return np.sum([f[(i,j)][x[i], x[j]] for i,j in E])
    fxstar = _eval(xstar)

    t = 0

    def smax(x, axis=None): return eps*sp.special.logsumexp(x/eps, axis)
    def sortt(t):
        if t[0] > t[1]:
            return (t[1], t[0])
        else:
            return t
    while t < B:
        # loop sets of dual variables
        for halpha in h:
            # copy inplace
            for edge in E:
                for i in range(2):
                    rhominus[(edge[i], edge[1-i])][:] = rho[(edge[i], edge[1-i])]
            # loop dual variables
            for (i,j) in halpha:
                d[i][:] = -np.inf
                for l in nbrs[i][0]:
                    r[i][:] = smax(f[(l, i)]-rhominus[(l, i)][:,None], axis=0) 
                    r[i] += smax(rhominus[(l, i)][:,None]-f[(l, i)], axis=0)
                    np.maximum(r[i], d[i], out=d[i])
                for l in nbrs[i][1]:
                    r[i][:] = smax(f[(i,l)]-rhominus[(l, i)][None,:], axis=1)
                    r[i] += smax(rhominus[(l, i)][None,:]-f[(i, l)], axis=1)
                    np.maximum(r[i], d[i], out=d[i])
                
                rho[(i,j)][:] = gamma[i]
                if i < j:
                    rho[(i,j)] += smax(f[(i,j)] - rhominus[(j,i)][None,:], axis=1)
                if j < i:
                    rho[(i,j)] += smax(f[(j,i)] - rhominus[(j,i)][:,None], axis=0)
                rho[(i,j)] -= d[i]           
        
        _opt()
        fx = _eval(x)
        if fx < fxstar:
            xstar[:] = x.copy()
            fxstar = fx

        t += 1

    # transform to original vertex names and edge orders
    orient(f, tp)
    rename(f, rn1)
    # rename edges in h
    hh = deepcopy(h)
    for s in range(len(h)):
        for edge_t in h[s]:
            hh[s].remove(edge_t)
            hh[s].add((rn1[edge_t[0]],rn1[edge_t[1]]))
    h = deepcopy(hh)
    xstar = {rn1[key]: xstar[key] for key in rn1.keys()}

    return xstar, fxstar

