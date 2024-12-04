import numpy as np
import itertools
import os
import pkg_resources
from ..optimize import bcadetr

def test_bcadetr_approx_inplace():
    n = 10
    m = 5
    k = np.array([7, 6, 3, 5, 7, 6, 3, 3, 4, 3])
    edges = np.array([[[9], [7], [7], [9], [8]],\
		   [[3], [4], [5], [7], [7]]])

    edges = [tuple(sorted([edges[0,v,0], edges[1,v,0]])) for v in range(edges.shape[1])]
    f1 = {edge: np.random.randn(k[edge[0]], k[edge[1]]) for edge in edges}
    f2 = {edge: f1[edge].copy() for edge in f1.keys()}

    bcadetr(f1, 1000)

    assert f1.keys() == f2.keys()
    assert np.all([np.allclose(f1[edge], f2[edge], rtol=1e-14, atol=1e-14) for edge in f1.keys()])

def test_bcadetr_tree():
    f = {(1,2): np.array([[3.,8,0],[6,8,9],[6,1,9]]),
         (2,3): np.array([[14.,5,7],[12,10,14],[12,5,7]]),
         (3,4): np.array([[7.,0,0],[1,1,0],[2,0,4]]),
         (3,5): np.array([[2.,3,0],[0,2,7],[6,0,0]])}
    c1 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    c2 = c1.copy()
    B = 10
    assert bcadetr(f, B, c=c1)==({1:0, 2:2, 3:1, 4:2, 5: 0}, 5.0)
    assert c1 == c2

def test_bcadetr_consistency():
    n = 100
    m = 2000
    k = np.array([69, 86, 96, 57, 97, 37, 45,  4, 10, 17, 18, 68, 95, 24, 16, 77, 47,
                    83, 50, 89, 54, 17, 99, 22, 33, 48, 53, 77, 52, 22,  3, 14, 13,  3,
                    85,  8,  8, 69, 62, 57, 41, 23, 58, 30,  9, 28,  9, 18, 26, 83, 47,
                    29, 62, 49, 37,  5, 83, 48,  7,  3, 69, 50, 42, 83, 57, 36, 17, 55,
                    99, 32, 40, 19, 88, 32, 35, 54, 47, 32, 83, 30, 94, 58, 95, 52, 97,
                    14, 85, 59,  9, 99, 93, 70, 60, 55, 98, 43, 74, 71, 93, 91])
    DATA_PATH = os.path.join(pkg_resources.resource_filename('pybiv', 'test/'), 'testdata.npy')
    edges = np.load(DATA_PATH)
    edges = [tuple(sorted([edges[0,v,0], edges[1,v,0]])) for v in range(edges.shape[1])]
    f1 = {edge: np.random.randn(k[edge[0]], k[edge[1]]) for edge in edges}
    f2 = f1.copy()
    x, fx = bcadetr(f1, 2)
    assert np.isclose(fx, np.sum([f2[(i,j)][x[i], x[j]] for i,j in f2.keys()]),\
                        rtol=1e-11, atol=1e-11)

def test_bcadetr_accuracy():
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
    x, fx = bcadetr(f, 10)
    assert np.isclose(fxstar, fx, rtol=1e-11, atol=1e-11)

