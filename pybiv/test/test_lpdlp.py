import numpy as np
import itertools
import os
import pkg_resources
from ..optimize import lpdlp

def test_lpdlp_approx_inplace():
    n = 10
    m = 5
    k = np.array([7, 6, 3, 5, 7, 6, 3, 3, 4, 3])
    edges = np.array([[[9], [7], [7], [9], [8]],\
		   [[3], [4], [5], [7], [7]]])

    edges = [tuple(sorted([edges[0,v,0], edges[1,v,0]])) for v in range(edges.shape[1])]
    f1 = {edge: np.random.randn(k[edge[0]], k[edge[1]]) for edge in edges}
    f2 = {edge: f1[edge].copy() for edge in f1.keys()}

    lpdlp(f1)

    assert f1.keys() == f2.keys()
    assert np.all([np.allclose(f1[edge], f2[edge], rtol=1e-14, atol=1e-14) for edge in f1.keys()])

def test_lpdlp_tree():
    f = {(1,2): np.array([[3.,8,0],[6,8,9],[6,1,9]]),
         (2,3): np.array([[14.,5,7],[12,10,14],[12,5,7]]),
         (3,4): np.array([[7.,0,0],[1,1,0],[2,0,4]]),
         (3,5): np.array([[2.,3,0],[0,2,7],[6,0,0]])}
    c1 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    c2 = c1.copy()
    B = 10
    assert lpdlp(f, c=c1)==({1:0, 2:2, 3:1, 4:2, 5: 0}, 5.0)
    assert c1 == c2

def test_lpdlp_consistency():
    np.random.seed(0)
    n = 16
    K = np.random.randint(2, 3, n)
    edges = [(0, 8), (3, 4), (0, 6), (11, 14), (1, 15), (2, 7), (12, 14), (9, 10), (5, 14), (3, 13)]
    f1 = {edge: np.random.randn(K[edge[0]], K[edge[1]]) for edge in edges}
    f2 = f1.copy()
    x, fx = lpdlp(f1)
    assert np.isclose(fx, np.sum([f2[(i,j)][x[i], x[j]] for i,j in f2.keys()]),\
                        rtol=1e-11, atol=1e-11)

def test_lpdlp_accuracy():
    np.random.seed(0)
    n = 16
    K = np.random.randint(2, 3, n)
    edges = [(0, 8), (3, 4), (0, 6), (11, 14), (1, 15), (2, 7), (12, 14), (9, 10), (5, 14), (3, 13)]
    f = {edge: np.random.randn(K[edge[0]], K[edge[1]]) for edge in edges}

    fxstar = np.inf
    for x in itertools.product(*list(map(range, K))):
        fx = np.sum([f[(i,j)][x[i], x[j]] for i,j in f.keys()])
        if fx < fxstar:
            fxstar = fx
    x, fx = lpdlp(f)
    assert np.isclose(fxstar, fx, rtol=2e-1, atol=9e-1)

