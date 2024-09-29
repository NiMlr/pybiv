import numpy as np


def orient(f, tp = None):
    """
    Orient the indices of a sum of bivariate functions.

    Given a sum of bivariates represented as a 
    dictionary with integer tuples as keys and 2-d `numpy.ndarray`
    as values, this function sorts each tuple increasingly
    and transposes the arrays of those tuples that were unsorted
    increasingly.

    If the parameter `tp` is given, then the function
    swaps the order of each tuple in the keys of f that is also in `tp`
    and transposes their values. 

    Parameters
    ----------
    f : dict
        Dictionary with unique integer tuples as keys and 2-d
        `numpy.ndarray`s as values. Represents bivariate
        functions. Will be returned inplace with some of its
        keys reordered and values transposed.
    tp : set, optional
        Subset of `set(f.keys())`.
        If passed, represents key-value-pairs to be reordered/transposed.
        The other key-value pairs will then remain unchanged.
        If not passed, instead keys of `f` will be sorted increasingly and
        values of changed keys will be transposed.

    Returns
    -------
    set
        Keys of changed key-value-pairs of `f`.
        The output will be a subset of `set(f.keys())` after
        completion.

    Raises
    ------
    ValueError
        If `f.keys()` contains tuples that have the same values
        in different order. Useful for inverting this function.
    """
    E = set(f.keys())
    if tp is None:
        tpn = set()

    while len(E) > 0:
        edge = E.pop()

        if (edge[1], edge[0]) in E:
            raise ValueError("The variable dependency graph \
                            can not have edges in both directions!")

        if tp is not None and edge in tp:
            tp.add((edge[1], edge[0]))
            tp.remove(edge)
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
    """
    Rename the indices of a sum of bivariate functions.

    The indices are represented by integer tuples, which
    are the keys of a passed dictionary.
    
    By default, the integers in the
    tuples will be replaced by other integers such that order
    is preserved, zero is the smallest and the maximum is the
    smallest possible.

    If `rn` is passed, each integer in the keys of `f` will be
    mapped to the corresponding value that is assingned to this
    integer by `rn`.

    The function works inplace and by recursion avoids clashes
    when mapping keys.

    Parameters
    ----------
    f : dict
        Dictionary with unique integer tuples as keys.
        Represents the indices of bivariate functions.
        Will be returned inplace with some of its keys renamed.
    rn : dict, optional
        `rn.keys()` must be `{v for key in set(f.keys()) for v in key}`.
        Values must be integers unique for each key.

    Returns
    -------
    dict
        `rn.keys()` are `{v for key in set(f.keys()) for v in key}`.
        Maps each integer value to the integer value it had before
        calling this function. Useful for inverting this function.
    """
    E = set(f.keys())
    if rn is None:
        # sorted list of vertices
        rn = sorted(list({v for key in E for v in key}))
        # map vertex to index
        rn = {rn[i]:i for i in range(len(rn))}
    
    while len(E) > 0:
        key = E.pop()
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
    """
    Order the arguments of a sum of bivariate functions
    by the number of functions containing the argument.

    Parameters
    ----------
    f : dict
        Dictionary with unique integer tuples as keys.
        Smallest integer to appear must be `0`.
        Every integer smaller than the largerst and larger
        than `0` must appear.
        Keys represent the indices of bivariate functions.

    Returns
    -------
    np.ndarray
        Length of the array is the number of unique
        integers in the keys of `f`.
        Maps each of its indices to a rank.
        An index with a smaller rank than another indicates
        that a larger or equal number of keys in
        `f.keys()` contain this index.
    np.ndarray
        `np.argsort(c)`, where `c` is the other returned array.
    """
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
    """
    Find smaller and larger neighbors in a directed graph.

    Parameters
    ----------
    E : set
        Set of integer tuples representing the edges of a
        directed graph. The integers represent the vertex
        indices. 
    n : int
        Largest vertex index of the graph.

    Returns
    -------
    list
        List of tuples, where each of the tuples contains two sets.
        The list is indexed by the vertex index.
        The first element in each tuple contains the indices of the
        smaller neighbors and the second element contains the indices
        of the larger neighbors. 
    """
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
