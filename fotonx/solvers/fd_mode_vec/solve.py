import numpy as np
from scipy.sparse.linalg import eigs

from fotonx.core.reshape import to_vec, to_mat
from fotonx.core.operators_fd import build_2d_operators
from .types import VModeResult, EpsTensor2D, MuTensor2D

def solve_vec_modes(A, wvl: float, Ny: int, Nx: int,
                    num_modes: int = 2,
                    n_guess: float = 1.45,
                    beta_guess: float | None = None,
                    z_fields: bool = False,
                    eps_used: EpsTensor2D | None = None,
                    mu_used: MuTensor2D | None = None,
                    dx: float | None = None,
                    dy: float | None = None,
                    ops: dict | None = None) -> VModeResult:

    k0 = 2*np.pi / wvl
    omega =  k0
    if beta_guess is None:
        beta_guess = k0 * float(n_guess)

    vals, vecs = eigs(A, k=num_modes, sigma=beta_guess)   # vecs: (4*N, k)

    order = np.argsort(np.abs(vals - beta_guess))
    vals = vals[order]
    vecs = vecs[:, order]

    N = Ny * Nx
    Ex_list, Ey_list, Hx_list, Hy_list = [], [], [], []
    Ez_list, Hz_list = ([] , []) if z_fields else (None, None)

    if z_fields:
        if eps_used is None or mu_used is None:
            raise ValueError("z_fields=True requires eps_used and mu_used")
        if ops is None:
            if dx is None or dy is None:
                raise ValueError("z_fields=True requires (dx,dy) or prebuilt ops")
            ops = build_2d_operators(Nx, Ny, dx, dy)
        Vx2, Vy2 = ops["Vx2"], ops["Vy2"]
        omega = 2*np.pi / wvl
        j = 1j

        eps_zz = to_vec(eps_used.eps_zz)
        eps_xz = to_vec(eps_used.eps_xz)
        eps_yz = to_vec(eps_used.eps_yz)
        mu_zz  = to_vec(mu_used.mu_zz)

    for i in range(num_modes):
        F = vecs[:, i]

        Ex = to_mat(F[0*N:1*N], Ny, Nx)
        Ey = to_mat(F[1*N:2*N], Ny, Nx)
        Hx = to_mat(F[2*N:3*N], Ny, Nx)
        Hy = to_mat(F[3*N:4*N], Ny, Nx)

        Ex_list.append(Ex); Ey_list.append(Ey)
        Hx_list.append(Hx); Hy_list.append(Hy)

        if z_fields:
            Exv, Eyv = to_vec(Ex), to_vec(Ey)
            Hxv, Hyv = to_vec(Hx), to_vec(Hy)

            Ezv = ((Vx2 @ Hyv - Vy2 @ Hxv)/(j*omega) - eps_xz*Exv - eps_yz*Eyv) / eps_zz
            Hzv = (Vy2 @ Exv - Vx2 @ Eyv) / (j*omega*mu_zz)

            Ez_list.append(to_mat(Ezv, Ny, Nx))
            Hz_list.append(to_mat(Hzv, Ny, Nx))

    beta = vals
    neff = beta / k0

    return VModeResult(beta=beta, neff=neff,
                       Ex=Ex_list, Ey=Ey_list, Hx=Hx_list, Hy=Hy_list,
                       Ez=Ez_list, Hz=Hz_list)



