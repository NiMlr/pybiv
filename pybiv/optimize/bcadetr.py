import numpy as np
import scipy as sp
from copy import deepcopy
from ..tools import (orient, rename, get_c, get_nbrs)


def bcadetr(f, B, c=None, eps=.1):
    """Try to minimize a sum of bivariate functions.

    Try to solve the NP-complete problem of minimizing
    a sum of discrete bivariate functions with a
    legacy implementation of the BCADETR method.
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

    eps : float
        Entropy regularization constant. Must be positive float.
        Default is `.1`.

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
    
    mmhi = {(edge[i], edge[1-i]): np.zeros(K[edge[i]]) for edge in E for i in range(2)}
    mmlo = {(edge[i], edge[1-i]): np.zeros(K[edge[i]]) for edge in E for i in range(2)}
    rho = {(edge[i], edge[1-i]): np.zeros(K[edge[i]]) for edge in E for i in range(2)}
    m = {i: np.zeros(K[i]) for i in range(n)}
    r = {i: np.zeros(K[i]) for i in range(n)}

    
    # count number of neighbors and collect smaller and larger neighbors
    nbrs = get_nbrs(E, n)
    
    nn = [len(nbrs[i][0])+len(nbrs[i][1]) for i in range(n)]
    gam = [eps*nn[i] + eps*nn[i]*np.log(nn[i]/eps) for i in range(n)]
    M = {i: np.zeros(K[i]*nn[i]) for i in range(n)}
    rc = np.array([0.])

    # (candidate) solutions, optimization, evaluations
    x = np.zeros(n, dtype=int)
    xstar = np.zeros(n, dtype=int)
    def _opt():
    # manipulates g, r and mmhi
        for i in range(n):
            r[i][:] = 0.
            for j in nbrs[i][1]:
                g[(i,j)][:] = f[(i,j)]
                g[(i,j)] -= rho[(j,i)][None,:]
                np.min(g[(i,j)], axis=1, out=mmhi[(i,j)])
                r[i][:] += mmhi[(i,j)]
            for j in nbrs[i][0]:
                r[i][:] += f[(j,i)][x[j], :]
            x[i] = np.argmin(r[i])

    def _eval(x):
        return np.sum([f[(i,j)][x[i], x[j]] for i,j in E])
    fxstar = _eval(xstar)

    def smax(x, out, axis=None):
    # inplace eps*logsumexp(./exp) along axis
    # writes to out
    # destroys x
        x.sort(axis=axis)
        x /= eps
        np.max(x, axis=axis, out=out)
        if axis == 1:
            x[:] -= out[:,None]
            np.exp(x, out=x)
            np.sum(x[:,:-1], axis=axis, out=x[:,0])
            np.log1p(x[:,0], out=x[:,0])
            out +=x[:,0]
        elif axis == 0:
            x[:] -= out[None,:]
            np.exp(x, out=x)
            np.sum(x[:-1,:], axis=axis, out=x[0,:])
            np.log1p(x[0,:], out=x[0,:])
            out += x[0,:]
        out *= eps


    t = 0
    g = deepcopy(f)

    while t < B:
        # loop vertices in order defined by c /reversed c (we reordered them)
        for i in range(n):
            # accumulate
            m[i][:] = 0.
            # loop "increasing"/"decreasing" edges wrt to c
            for j in nbrs[i][1]:
                # inplace
                g[(i,j)][:] = f[(i,j)]
                g[(i,j)] -= rho[(j,i)][None,:]
                smax(g[(i,j)], mmhi[(i,j)], axis=1)
                m[i] += mmhi[(i,j)]

                g[(i,j)][:] = rho[(j,i)][None,:]
                g[(i,j)] -= f[(i,j)]
                smax(g[(i,j)], mmlo[(i,j)], axis=1)
                mmlo[(i,j)] *= -1.


            for j in nbrs[i][0]:
                # inplace
                g[(j,i)][:] = f[(j,i)]
                g[(j,i)] -= rho[(j,i)][:,None]
                smax(g[(j,i)], mmhi[(i,j)], axis=0)
                m[i] += mmhi[(i,j)]

                g[(j,i)][:] = rho[(j,i)][:,None]
                g[(j,i)] -= f[(j,i)]
                smax(g[(j,i)], mmlo[(i,j)], axis=0)
                mmlo[(i,j)] *= -1.

            m[i] /= nn[i]

            # compute constant rc
            jc = 0
            for j in nbrs[i][1]:
                M[i][jc*K[i]:(jc+1)*K[i]] = mmhi[(i,j)]
                M[i][jc*K[i]:(jc+1)*K[i]] -= mmlo[(i,j)]
                M[i][jc*K[i]:(jc+1)*K[i]] -= m[i]
                jc += 1
            for j in nbrs[i][0]:
                M[i][jc*K[i]:(jc+1)*K[i]] = mmhi[(i,j)]
                M[i][jc*K[i]:(jc+1)*K[i]] -= mmlo[(i,j)]
                M[i][jc*K[i]:(jc+1)*K[i]] -= m[i]
                jc += 1

            smax(M[i][:,None], rc, axis=0)
            rc = gam[i]-nn[i]*rc
            
            # distribute
            for j in nbrs[i][1]:
                rho[(i,j)][:] = mmhi[(i,j)]
                rho[(i,j)] += rc/nn[i]
                rho[(i,j)] -= m[i]
            for j in nbrs[i][0]:
                rho[(i,j)][:] = mmhi[(i,j)]
                rho[(i,j)] += rc/nn[i]
                rho[(i,j)] -= m[i]
                
        
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

