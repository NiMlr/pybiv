import numpy as np
import scipy as sp
from ..tools import (orient, rename, get_c, get_nbrs)


def trws(f, B, c=None):
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
    m = {i: np.zeros(K[i]) for i in range(n)}

    

    
    # count number of neighbors and collect smaller and larger neighbors
    nbrs = get_nbrs(E, n)
    
    p = [max(len(nbrs[v][0]), len(nbrs[v][1])) for v in range(n)]

    gam = {(edge[i], edge[1-i]): 1./p[i] for edge in E for i in range(2)}

    # (candidate) solutions, optimization, evaluations
    x = np.zeros(n, dtype=int)
    xstar = np.zeros(n, dtype=int)
    def _opt():
        for i in range(n):
            m[i][:] = 0.
            for j in nbrs[i][1]:
                m[i][:] += mm[(i,j)]
            for j in nbrs[i][0]:
                m[i][:] += f[(j,i)][x[j], :]
            x[i] = np.argmin(m[i])

    def _eval(x):
        return np.sum([f[(i,j)][x[i], x[j]] for i,j in E])
    fxstar = _eval(xstar)

    t, delta = 0, 0
    # for reversing without reindexing
    # reverses all c-depended parts separately
    def _revon(y):
        if isinstance(y, range):
            return reversed(y)
        elif isinstance(y, int):
            return 1-y
        elif isinstance(y, tuple):
            return (y[1], y[0])
        elif isinstance(y, np.ndarray):
            return y.transpose()
    _rev = lambda y: y


    while t < B:
        # loop vertices in order defined by c /reversed c (we reordered them)
        for i in _rev(range(n)):
            for j in nbrs[i][0]:
                m[i] += mm[(i,j)]
            for j in nbrs[i][1]:
                m[i] += mm[(i,j)]
            
            delta = np.min(m[i])
            m[i] -= delta

            # loop "increasing"/"decreasing" edges wrt to c
            for j in nbrs[i][_rev(1)]:
                # inplace
                m[i] *= gam[(i,j)]
                f[_rev((i,j))] += _rev(m[i][:,None])
                f[_rev((i,j))] -= _rev(mm[(i,j)][:,None])
                np.min(f[_rev((i,j))], axis=_rev(0), out=mm[(j,i)])
                f[_rev((i,j))] += _rev(mm[(i,j)][:,None])
                f[_rev((i,j))] -= _rev(m[i][:,None])
                m[i] /= gam[(i,j)]
                
                delta = np.min(mm[(j,i)])
                mm[(j,i)] -= delta
        
        _opt()
        fx = _eval(x)
        if fx < fxstar:
            xstar[:] = x.copy()
            fxstar = fx

        t += 1
        if t%2 == 1:
            _rev = _revon
        else:
            _rev = lambda y: y

    # transform to original vertex names and edge orders
    orient(f, tp)
    rename(f, rn2)
    rename(f, rn1)
    c = {rn1[i]: c[i] for i in range(len(rn1))}

    xstar = {rn2[i]: xstar[i] for i in range(len(rn2))}
    xstar = {rn1[key]: xstar[key] for key in xstar.keys()}

    return xstar, fxstar

