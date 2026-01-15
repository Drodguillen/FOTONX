import numpy as np
from scipy.sparse import diags, kron, eye

def U(N:int, d:float):
    main = -1.0 * np.ones(N)
    off = 1.0 * np.ones(N)
    return diags([main,off], offsets=[0,1], shape=(N,N), format="csr")/d

def V(N:int, d:float):
    main = 1.0 * np.ones(N)
    off = -1.0 * np.ones(N)
    return diags([off,main], offsets=[-1,0], shape=(N,N), format="csr")/d

def build_2d_operators(Nx:int, Ny:int, dx:float, dy:float):
    """
    Returns a dict of sparse operators on a collocated 2D grid:
    Ux2, Uy2, Vx2, Vy2: one-sided derivatives
    Dx, Dy: central first derivatives
    Dxx, Dyy: second derivatives
    Dxy, Dyx: mixed derivatives
    """

    Ux = U(Nx, dx)
    Uy = U(Ny, dy)
    Vx = V(Nx, dx)
    Vy = V(Ny, dy)

    Ix = eye(Nx, format="csr")
    Iy = eye(Ny, format="csr")

    Ux2 = kron(Iy, Ux, format="csr")
    Uy2 = kron(Uy, Ix, format="csr")
    Vx2 = kron(Iy, Vx, format="csr")
    Vy2 = kron(Vy, Ix, format="csr")

    Dx = 0.5 * (Vx2 + Ux2)
    Dy = 0.5 * (Vy2 + Uy2)

    Dxx = Vx2 @ Ux2
    Dyy = Vy2 @ Uy2

    Dxy = Dy @ Dx
    Dyx = Dx @ Dy

    return dict(
            Ux2=Ux2, Uy2=Uy2, Vx2=Vx2, Vy2=Vy2,
            Dx=Dx, Dy=Dy, Dxx=Dxx, Dyy=Dyy, Dxy=Dxy, Dyx=Dyx
            )
