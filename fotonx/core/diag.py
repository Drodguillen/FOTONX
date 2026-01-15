import numpy as np
from scipy.sparse import diags

def diag_from(arr2d: np.ndarray):
    """
    return CSR diagonal matrix of arr2d flattenned in c-order
    """
    v = arr2d.ravel(order="C")
    N = v.size
    return diags(v, 0, shape=(N,N), dtype=complex, format="csr")

