import numpy as np
import scipy as sp
import itertools
    

def approx(F, K=None):
    if K is None:
        K = F.shape
        handle = False
    else:
        handle = True
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
    err = np.linalg.norm(A@sol[0]-B)
    res = (A@sol[0]-B).reshape(K, order='C')
    f_biv = reshape_biv(sol[0], K)
    return (f_biv, err, res)

def reshape_biv(biv_vector, K):
    n = len(K)
    f_biv = {}
    var_ind = 0
    for pair in itertools.combinations(range(n),2):
        succ = var_ind+K[pair[0]]*K[pair[1]]
        f_biv[pair] = biv_vector[var_ind:succ].reshape(K[pair[0]],K[pair[1]])
        var_ind = succ
    return (f_biv, K)

