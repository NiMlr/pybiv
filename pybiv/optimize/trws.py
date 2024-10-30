import numpy as np
import scipy as sp
from copy import deepcopy
from ..tools import (orient, rename, get_c, get_nbrs)


def trws_leg(f, B, c=None):
    """Try to minimize a sum of bivariate functions.

    Try to solve the NP-complete problem of minimizing
    a sum of discrete bivariate functions with a
    legacy implementation of the TRW-S method.
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

    References
    ----------
    .. [VK] Kolmogorov, Vladimir, 2005, "Convergent tree-reweighted message
            passing for energy minimization," *Proceedings of Machine Learning Research*
            R5: 182-189.
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
    g = deepcopy(f)
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
                g[_rev((i,j))][:] = f[_rev((i,j))]
                g[_rev((i,j))] += _rev(m[i][:,None])
                g[_rev((i,j))] -= _rev(mm[(i,j)][:,None])
                np.min(g[_rev((i,j))], axis=_rev(0), out=mm[(j,i)])
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


def trws(f, B, c=None):
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

    References
    ----------
    .. [VK] Kolmogorov, Vladimir, 2005, "Convergent tree-reweighted message
            passing for energy minimization," *Proceedings of Machine Learning Research*
            R5: 182-189.
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
    
    p = [max(len(nbrs[v][0]), len(nbrs[v][1])) for v in range(n)]

    gam = {(edge[i], edge[1-i]): 1./p[i] for edge in E for i in range(2)}

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
    g = deepcopy(f)
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
            m[i][:] = 0.
            for j in nbrs[i][0]:
                m[i] += mm[(i,j)]
            for j in nbrs[i][1]:
                m[i] += mm[(i,j)]

            # loop "increasing"/"decreasing" edges wrt to c
            for j in nbrs[i][_rev(1)]:
                # inplace
                g[_rev((i,j))][:] = f[_rev((i,j))]
                g[_rev((i,j))] -= _rev(rho[(j,i)][None,:])
                np.min(g[_rev((i,j))], axis=_rev(1), out=mm[(i,j)])

                # inplace
                rho[(i,j)][:] = mm[(i,j)]
                m[i] *= gam[(i,j)]
                rho[(i,j)] -= m[i]
                m[i] /= gam[(i,j)]
                
        
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

