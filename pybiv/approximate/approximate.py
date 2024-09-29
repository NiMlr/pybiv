import numpy as np
import scipy as sp
import itertools
    

def approx(F, K=None):
    """l^2-Best approximate a discrete function by a sum of bivarite functions.

    Parameters
    ----------
    F : np.ndarray or callable
        If of type `np.ndarray` represents an evaluated function to the reals.
        `F.shape` contains the number discrete values of each factor of the domain.
        Its values must be numeric.
 
        If of type `callable`, then `F` takes as one argument a tuple of integers.
        The length of the accepted tuple must be the length of `K`.
        `F` must accept integer tuples where the `i`-th integer can range
        from `0` to `K[i]-1`.

    K : tuple, optional
        Only needed if `F` is a callable. Otherwise `K == F.shape`.
        Tuple of positive integers containing
        the number discrete values of each factor of the domain of `F`.

    Returns
    -------
    tuple
        Contains as a first element a dictionary where keys are tuples of
        two integers and values are 2-d float `np.ndarray`s.
        For integer `i` contained in some key `(i, j)` or `(j, i)`,
        we associate the first or second element of the shape of the value
        with the number of discrete values of each factor of the functions
        domain. It is consistent over all key's integers values.
        Therefore, the first element represents dictionary of (compatible)
        bivariate functions.

        The second element is a tuple itself containing the numbers of
        discrete values of each factor of the bivariates domains indexed
        by the same integer values that the keys of the dictionary contain.
        It resembles the input argument `K`.
    
    float
        The l^2-norm of the approximation residual.

    np.ndarray
        The approximation residual.
        Its shape contains the number of discrete values of each factor of
        the domain. It resembles the input argument `K`.
        Its values are floats.
    """
    if F is callable:
        handle = True
    else:
        K = F.shape
        handle = False
    n = len(K)

    num_vars_t = 0
    for pair in itertools.combinations(list(range(n)),2):
        num_vars_t += K[pair[0]]*K[pair[1]]
    num_eqs_t = np.prod(K)

    if handle:
        B = np.zeros(num_eqs_t)

    I = np.empty(num_eqs_t*n*(n-1)//2, dtype=int)
    J = np.empty(num_eqs_t*n*(n-1)//2, dtype=int)

    eq_ind = -1
    ind_t = -1
    for x in itertools.product(*list(map(range, K))):
        eq_ind += 1
        var_ind = 0
        # iterate bivariate functions
        for pair in itertools.combinations(range(n),2):
            ind_t += 1
            # set cur_ind // row major
            cur_var_ind = var_ind + x[pair[0]]*K[pair[0]] + x[pair[1]]
            # set entry
            I[ind_t] = eq_ind
            J[ind_t] = cur_var_ind
            # add total number of vars
            var_ind += K[pair[0]]*K[pair[1]]

        if handle:
            B[eq_ind] = F(x)

    if not handle:
        B = F.flatten()

    A = sp.sparse.coo_array((np.ones(num_eqs_t*n*(n-1)//2, dtype=int), (I, J)), shape=(num_eqs_t, num_vars_t))
    A.tocsc()


    sol = sp.sparse.linalg.lsmr(A,B, atol=1e-14, btol=1e-14)
    res = (A@sol[0]-B).reshape(K, order='C')
    err = np.linalg.norm(res)
    f_biv = _reshape_biv(sol[0], K)
    return (f_biv, err, res)

def _reshape_biv(biv_vector, K):
    n = len(K)
    f_biv = {}
    var_ind = 0
    for pair in itertools.combinations(range(n),2):
        succ = var_ind+K[pair[0]]*K[pair[1]]
        f_biv[pair] = biv_vector[var_ind:succ].reshape(K[pair[0]],K[pair[1]])
        var_ind = succ
    return (f_biv, K)
