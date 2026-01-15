from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

def cn_step(F, A, dx:float):
    """
    One Crank-Nicolson step fro dF/dz = A F
    Work for any sparse A(N x N) and vector F(N,)
    """

    I = eye(A.shape[0], format="csr", dtype=complex)
    M_left = (I - 0.5 * dz * A)
    M_right = (I + 0.5 * dz * A)
    rhs = M_right @ F
    return spsolve(M_left, rhs)


