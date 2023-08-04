from itertools import product

import numpy as np


def pareto(x):
    """
    x: Array with dims (n, m).
    Returns: Indices of rows in x which are pareto optimal (lower = better).
    """
    n = x.shape[0]
    dom = np.empty((n, n), dtype=bool) # dom[i][j] = row i pareto dominates row j
    for i, j in product(range(n), range(n)):
        dom[i][j] = all(x[i] < x[j])
    return [j for j in range(n) if not any(dom[:, j])]

def rawlsian(x):
    """
    x: Array with dims (n, m).
    Returns: Indices of rows chosen by Ralwsian choice (lowest maximum).
    """
    maxes = x.max(axis=1)
    return np.where(maxes == maxes.min())[0]

def choose(df):
    p = pareto(df.values)
    p_df = df.iloc[p]
    r = rawlsian(p_df.values) if len(p) else []
    return df.index[p], p_df.index[r]
