import numpy as np


def orient(f, tp = None):
    E = set(f.keys())
    if tp is None:
        tpn = set()

    while len(E) > 0:
        edge = E.pop()

        if (edge[1], edge[0]) in E:
            raise ValueError("The variable dependency graph \
                            can not have edges in both directions!")

        if tp is not None and edge in tp:
            pass
        elif tp is None and edge[0] > edge[1]:
            tpn.add((edge[1], edge[0]))
        else:
            continue
        f[(edge[1], edge[0])] = f[edge].transpose()
        del f[edge]
    
    if tp is None:
        tp = tpn
    return tp

def rename(f, rn=None):
    E = set(f.keys())
    if rn is None:
        # sorted list of vertices
        rn = sorted(list({v for key in E for v in key}))
        # map vertex to index
        rn = {rn[i]:i for i in range(len(rn))}
    
    while len(E) > 0:
        key = E.pop()
        if key == (1,12):
            aaa = 1
        _safe_switch(f, key, rn, E)
    
    rn = {rn[key]:key for key in rn.keys()}
    return rn

def _safe_switch(f, key, rn, E):
    new_key = (rn[key[0]], rn[key[1]])

    # will not be switched
    if new_key == key:
        return
    # new key exists as other
    elif new_key in E:
        temp = f[key]
        del f[key]
        _safe_switch(f, new_key, rn, E)
        E.remove(new_key)
        f[new_key] = temp
    else:
        f[new_key] = f[key]
        del f[key]

def get_c(f):
    E = set(f.keys())
    n = max({v for edge in E for v in edge})+1

    nnbrs = np.zeros(n, dtype=int)

    # init count number of neighbors
    for edge in E:
        v1, v2 = edge[0], edge[1]
        nnbrs[v1] += 1
        nnbrs[v2] += 1

    c = np.max(nnbrs)-nnbrs
    cinv = np.argsort(c)
    c[:] = np.argsort(cinv)
    return c, cinv

def get_nbrs(E, n):
    # init count number of neighbors and collect smaller and larger neighbors
    nbrs = [(set(), set()) for v in range(n)]

    # collect smaller and larger neighbors
    for edge in E:
        v1, v2 = edge[0], edge[1]
        if v1 < v2:
            nbrs[v2][0].add(v1)
            nbrs[v1][1].add(v2)
        else:
            nbrs[v2][1].add(v1)
            nbrs[v1][0].add(v2)
    
    return nbrs

