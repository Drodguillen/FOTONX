import numpy as np
from scipy.sparse import diags, kron, eye, bmat
from fotonx.core.diag import diag_from
from .types import EpsTensor2D, MuTensor2D
from fotonx.core.operators_fd import build_2d_operators

def assemble_mts(eps_t: EpsTensor2D, mu_t: MuTensor2D, Nx, Ny, dx, dy, wvl):
    """
    Docstring for assemble_mts
    
    eps_t: Epsilon Tensor
    Nx: Number of cells in x
    Ny: Number of cells in y
    dx: x space
    dy: y space
    wvl: Wavelength
    """
    omega = 2*np.pi/wvl
    N = Nx*Ny
    I_N = eye(N, format="csr", dtype=complex) # type: ignore
    j = 1j

    ops = build_2d_operators(Nx, Ny, dx, dy)
    Ux2, Vx2, Uy2, Vy2, Dx, Dy, Dxx, Dyy = ops["Ux2"], ops["Vx2"], ops["Uy2"], ops["Vy2"], ops["Dx"], ops["Dy"], ops["Dxx"], ops["Dyy"]


    A11 = -j*diag_from(eps_t.eps_xz / eps_t.eps_zz)*Ux2
    A12 = -j*diag_from(eps_t.eps_yz / eps_t.eps_zz)*Ux2
    A13 = - diag_from(1 / (omega * eps_t.eps_zz))*Ux2*Vy2
    A14 = diag_from(1 / (omega * eps_t.eps_zz))*Ux2*Vx2 + omega*diag_from(mu_t.mu_yy)*I_N

    A21 = -j*diag_from(eps_t.eps_xz / eps_t.eps_zz)*Uy2
    A22 = -j*diag_from(eps_t.eps_yz / eps_t.eps_zz)*Uy2
    A23 = - diag_from(1 / (omega * eps_t.eps_zz))*Uy2*Vy2 - omega*diag_from(mu_t.mu_xx)*I_N
    A24 = diag_from(1 / (omega * eps_t.eps_zz))*Uy2*Vx2

    A31 = -omega*diag_from(eps_t.eps_xy)*I_N + diag_from(1 / (omega*mu_t.mu_zz))*Vx2*Uy2 + diag_from((eps_t.eps_yz*omega*eps_t.eps_xz) / eps_t.eps_zz)*I_N
    A32 = -omega*diag_from(eps_t.eps_yy)*I_N - diag_from(1 / (omega*mu_t.mu_zz))*Vx2*Ux2 + diag_from((eps_t.eps_yz*omega*eps_t.eps_yz) / eps_t.eps_zz)*I_N
    A33 = -j*diag_from(eps_t.eps_yz / eps_t.eps_zz)*Vy2
    A34 = j*diag_from(eps_t.eps_yz / eps_t.eps_zz)*Vx2

    A41 = omega*diag_from(eps_t.eps_xx)*I_N + diag_from(1 / (omega*mu_t.mu_zz))*Vy2*Uy2 - diag_from((eps_t.eps_xz*omega*eps_t.eps_xz) / eps_t.eps_zz)*I_N
    A42 = omega*diag_from(eps_t.eps_xy)*I_N - diag_from(1 / (omega*mu_t.mu_zz))*Vy2*Ux2 - diag_from((eps_t.eps_xz*omega*eps_t.eps_yz) / eps_t.eps_zz)*I_N
    A43 = j*diag_from(eps_t.eps_xz / eps_t.eps_zz)*Vy2
    A44 = -j*diag_from(eps_t.eps_xz / eps_t.eps_zz)*Vx2

    A = bmat([[A11, A12, A13, A14],
              [A21, A22, A23, A24],
              [A31, A32, A33, A34],
              [A41, A42, A43, A44]], format="csr")
    
    return A