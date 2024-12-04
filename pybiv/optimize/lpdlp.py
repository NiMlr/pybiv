import numpy as np
import scipy as sp
import itertools
import sys
from ..tools import (orient, rename, get_c, get_nbrs)
np.set_printoptions(threshold=sys.maxsize)


def lpdlp(f, c=None):
    """Try to minimize a sum of bivariate functions.

    Try to solve the NP-complete problem of minimizing
    a sum of discrete bivariate functions with linear programming
    for the dual linear program.
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

    E = list(f.keys())
    n = max({v for edge in E for v in edge})+1

    # variable cardinality (Omega)
    K = {edge[i]:f[edge].shape[i] for edge in E for i in range(2)}
    K = [K[v] for v in range(n)]

    # count number of neighbors and collect smaller and larger neighbors
    nbrs = get_nbrs(E, n)

    # dictionary mapping to the indices of rho
    ind_rho = {}
    rho_len = 0
    for v in range(n):
        ind_rho[v] = (rho_len, rho_len + 1)
        rho_len += 1
    for edge in E:
        ind_rho[edge] = (rho_len, rho_len + K[edge[0]])
        rho_len += K[edge[0]]
        ind_rho[(edge[1],edge[0])] = (rho_len, rho_len + K[edge[1]])
        rho_len += K[edge[1]]

    # build equality
    b_eq_len = np.sum(K)
    b_eq = np.zeros(b_eq_len)

    I_eq = []
    J_eq = []
    D_eq = []
    i = 0
    for v in reversed(range(n)):
        # K[v] equations for each v
        for xv in range(K[v]):
            # number of neighbors
            nn = len(nbrs[v][0])+len(nbrs[v][1])
            # in the same eq nn + 1 variables
            I_eq.extend([i]*(nn+1))
            # variable index of v plus index of xv
            J_eq.append(ind_rho[v][0])
            for nbr in nbrs[v][0]:
                J_eq.append(ind_rho[(v, nbr)][0]+xv)
            for nbr in nbrs[v][1]:
                J_eq.append(ind_rho[(v, nbr)][0]+xv)

            D_eq.append(1.)
            D_eq.extend([-1.]*nn)
            # next equation
            i += 1
    A_eq = sp.sparse.coo_array((D_eq, (I_eq, J_eq)), shape=(b_eq_len, rho_len))
    A_eq.tocsr()

    # dictionary mapping to the indices of the upper bound f
    ind_b_ub = {}
    b_ub_len = 0
    for i,j in E:
        ind_b_ub[(i,j)] = (b_ub_len, b_ub_len+K[i]*K[j])
        b_ub_len += K[i]*K[j]

    # build inequality
    b_ub = np.zeros(b_ub_len)
    for edge in E:
        ind = ind_b_ub[edge]
        b_ub[ind[0]:ind[1]] = f[edge].flatten()
        b_ub[ind[0]:ind[1]] -= np.min(f[edge].flatten())

    J_ub = []
    i = 0
    for (i,j) in E:
        # two variables each equation
        # row variable index is repeating
        ind = ind_rho[(i,j)]
        temp1 = np.arange(ind[0], ind[1])
        temp1 = np.repeat(temp1, K[j])

        # column variable index is tiling
        ind = ind_rho[(j, i)]
        temp2 = np.arange(ind[0], ind[1])
        temp2 = np.tile(temp2, K[i])
        # append them one after the other
        J_ub.extend(list(np.column_stack((temp1, temp2)).flatten()))

        # we added K[i]*K[j] equations
        i += K[i]*K[j]

    # two variables each equation
    I_ub = np.repeat(np.arange(b_ub_len), 2)
    D_ub = np.ones(len(J_ub))
    
    A_ub = sp.sparse.coo_array((D_ub, (I_ub, J_ub)), shape=(b_ub_len, rho_len))
    A_ub.tocsr()

    xstar = np.zeros(n, dtype=int)
    cost = np.zeros(rho_len)
    cost[:n] = -1.
    rho = np.zeros(rho_len)

    res = sp.optimize.linprog(cost, A_ub=A_ub, b_ub=b_ub,
                             A_eq=A_eq, b_eq=b_eq,
                             method="highs-ds")
                             #options={"presolve": False})

    rho[:] = res['x']
    # dual solution to primal candidate
    for i in range(n):
        r = np.zeros(K[i])
        for j in nbrs[i][1]:
            inds = ind_rho[(j,i)]
            r[:] += np.min(f[i,j] - rho[inds[0]:inds[1]][None,:], axis=1)
        for j in nbrs[i][0]:
            r[:] += f[(j,i)][xstar[j], :]
        xstar[i] = np.argmin(r)
    
    fxstar = np.sum([f[(i,j)][xstar[i], xstar[j]] for i,j in E])
    
    # transform to original vertex names and edge orders
    orient(f, tp)
    rename(f, rn2)
    rename(f, rn1)
    c = {rn1[i]: c[i] for i in range(len(rn1))}

    xstar = {rn2[i]: xstar[i] for i in range(len(rn2))}
    xstar = {rn1[key]: xstar[key] for key in xstar.keys()}

    return xstar, fxstar
