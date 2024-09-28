import numpy as np
from ..tools import (orient, rename, get_c, get_nbrs)


def test_rename_orient():
    f = {(3,4): np.array([[3,4]]),
          (4,7): np.array([[4],[5]]),
          (12,10): np.array([[12, 10]])}
    
    a = {(0,1): np.array([[3,4]]),
                (1,2): np.array([[4],[5]]),
                (3,4): np.array([[12, 10]]).transpose()}
    
    b = {(3,4): np.array([[3,4]]),
                  (4,7): np.array([[4],[5]]),
                  (12,10): np.array([[12, 10]])}
    
    # not inplace
    E = set(f.keys())
    # sorted list of vertices
    rn = sorted(list({v for key in E for v in key}))
    # map vertex to index
    rn = {rn[i]:i for i in range(len(rn))}

    f1 = {(rn[key[0]], rn[key[1]]): f[key] for key in f.keys()}
    rn = rename(f)
    assert f==f1
    tp = orient(f)
    assert f.keys() == a.keys()
    assert np.all([np.all(f[key] == a[key]) for key in f.keys()])
    rename(f)
    assert f.keys() == a.keys()
    assert np.all([np.all(f[key] == a[key]) for key in f.keys()])
    orient(f, tp)
    rename(f, rn)
    assert f.keys() == b.keys()
    assert np.all([np.all(f[key] == b[key]) for key in f.keys()])

def test_get_c():
    f = {(0,1): 0, (1,2): 1}
    assert list(get_c(f)[0]) == [1, 0, 2]
    f = {(0,1): 0, (0,2): 1}
    assert list(get_c(f)[1]) == [0, 1, 2]

def test_get_nbrs():
    E = {(0,1), (1,2), (3,4)}
    n = 5
    nbrs = get_nbrs(E, n)
    assert nbrs == [(set(), {1}), ({0}, {2}), ({1}, set()), (set(), {4}), ({3}, set())]

