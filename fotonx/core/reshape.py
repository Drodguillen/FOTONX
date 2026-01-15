import numpy as np

def to_vec(field2d: np.ndarray) -> np.ndarray:
    """
    (Ny, Nx) -> (N,) using row-major order.
    """
    return field2d.ravel(order="C")

def to_mat(vec: np.ndarray, Ny:int, Nx:int) -> np.ndarray:
    """
    (N,) -> (Nx, My) using row-major order
    """
    return vec.reshape((Ny,Nx), order="C")


