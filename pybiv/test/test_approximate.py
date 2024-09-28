from ..approximate import approx
import numpy as np
import itertools


def test_product():
    np.random.seed(42)
    B1 = np.random.rand(*np.random.randint(2, high=5, size=4))
    K = B1.shape
    B2 = np.empty(np.prod(B1.shape))
    i = -1
    for inp_args in itertools.product(*list(map(range, K))):
        i += 1
        B2[i] = B1[inp_args]
    
    assert np.array_equal(B1.flatten(order='C'), B2)
    assert np.array_equal(B1, B2.reshape(*K, order='C'))

def test_array_vs_function():
    def F_test(x):
        vals = [6.,10.,3.,0.,1.,3.,3.,1.,6.,6.,\
                10.,6.,1.,3.,0.,1.,6.,3.,1.,0.,\
                6.,1.,3.,15.,6.,3.,3.,10.,10.,\
                1.,3.,3.,6.,6.,6.,6.,1.,0.,1.,\
                6.,6.,3.,0.,1.,1.,1.,6.,3.,3.,\
                3.,0.,1.,10.,3.,3.,3.,6.,3.,1.,\
                3.,0.,3.,1.,10.]
        inp = int(''.join(map(str, x)), 2)
        return vals[inp] 

    F_test_array = np.empty((2,)*6)
    for x in itertools.product(*list(map(range, (2,)*6))):
        F_test_array[x] = F_test(x)

    app1 = approx(F_test, K=(2,)*6)
    app2 = approx(F_test_array)

    assert app1[0][0].keys() == app2[0][0].keys()
    for key in app1[0][0].keys():
        assert np.all(app1[0][0][key]==app2[0][0][key])

    assert app1[1]==app2[1]

def test_assembly():
    K = (2,2,2)
    n = len(K)

    num_vars_t = 0
    for pair in itertools.combinations(list(range(n)),2):
        num_vars_t += K[pair[0]]*K[pair[1]]
    num_eqs_t = np.prod(K)

    A = np.zeros((num_eqs_t, num_vars_t))

    eq_ind = -1
    for x in itertools.product(*list(map(range, K))):
        eq_ind += 1
        var_ind = 0
        # iterate bivariate functions
        for pair in itertools.combinations(range(n),2):
            # set cur_ind // row major
            cur_var_ind = var_ind + x[pair[0]]*K[pair[0]] + x[pair[1]]
            # set entry
            A[eq_ind, cur_var_ind] = 1
            # add total number of vars
            var_ind += K[pair[0]]*K[pair[1]]
    
    A_test = [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], #f01 00 + f02 00 + f12 00 # 0 0 0
              [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], #f01 00 + f02 01 + f12 01 # 0 0 1
              [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], #f01 01 + f02 00 + f12 10 # 0 1 0
              [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], #f01 01 + f02 01 + f12 11 # 0 1 1
              [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], #f01 10 + f02 10 + f12 00 # 1 0 0
              [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0], #f01 10 + f02 11 + f12 01 # 1 0 1
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], #f01 11 + f02 10 + f12 10 # 1 1 0
              [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], #f01 11 + f02 11 + f12 11 # 1 1 1
              ]
    assert np.all(A==A_test)


def test_convergence():
    def F_test(x):
        vals = [6.,10.,3.,0.,1.,3.,3.,1.,6.,6.,\
                10.,6.,1.,3.,0.,1.,6.,3.,1.,0.,\
                6.,1.,3.,15.,6.,3.,3.,10.,10.,\
                1.,3.,3.,6.,6.,6.,6.,1.,0.,1.,\
                6.,6.,3.,0.,1.,1.,1.,6.,3.,3.,\
                3.,0.,1.,10.,3.,3.,3.,6.,3.,1.,\
                3.,0.,3.,1.,10.]

        inp = int(''.join(map(str, x)), 2)
        return vals[inp]

    assert np.isclose(approx(F_test, K=(2,)*6)[1], 20.982135258357285)

