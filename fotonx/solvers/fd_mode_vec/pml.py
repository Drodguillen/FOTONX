import numpy as np
from .types import EpsTensor2D, MuTensor2D

def pml_um_to_px(npml_t_x, npml_t_y, dx, dy):
    """
    Conevrt micrometer thickness into cells
    
    npml_t_x: x pml thickness in um
    npml_t_y: y pml thickness in um
    """
    npml_x = int(round(npml_t_x/dx))
    npml_y = int(round(npml_t_y/dy))

    return npml_x, npml_y

def pml_profile(N, dx, npml, alpha_max):
    """
    Build a 1D stretch factor s_j = 1 - j*alpha, with
    alpha_j = alpha_max * (rho/d)^2 
    """

    s = np.ones(N, dtype=complex)
    if npml == 0 or alpha_max == 0.0:
        return s

    d = npml*dx
    for i in range(npml):
        rho = (npml - i)*dx
        alpha = alpha_max * (rho/d)**2
        s[i] = 1.0 - 1j*alpha

    for i in range(N-npml,N):
        rho = (i - (N - npml) + 1)*dx
        alpha = alpha_max * (rho/d)**2
        s[i] = 1.0 - 1j*alpha

    return s


def build_pml(Nx, Ny, dx, dy, npml_x, npml_y, alpha_x_max, alpha_y_max):
    """
    Build 2D stretch factors sx, sy, sz.
    """

    sx_1d = pml_profile(Nx,dx,npml_x,alpha_x_max)
    sy_1d = pml_profile(Ny,dy,npml_y,alpha_y_max)

    sx, sy = np.meshgrid(sx_1d, sy_1d, indexing="xy")
    sz = np.ones_like(sx, dtype=complex)

    return sx, sy, sz

def build_mu(sx, sy, sz):
    mu_x = (sy*sz/sx)
    mu_y = (sz*sx/sy)
    mu_z = (sx*sy/sz)

    return mu_x, mu_y, mu_z

def apply_pml_eps(eps: EpsTensor2D, sx, sy, sz):
    eps_xx_p = (sy*sz/sx) * eps.eps_xx
    eps_yy_p = (sx*sz/sy) * eps.eps_yy
    eps_zz_p = (sx*sy/sz) * eps.eps_zz

    eps_xy_p = sz * eps.eps_xy
    eps_yz_p = sx * eps.eps_yz
    eps_xz_p = sy * eps.eps_xz

    return EpsTensor2D(eps_xx_p, eps_yy_p, eps_zz_p,
                     eps_xy_p, eps_yz_p, eps_xz_p)

def apply_pml_mu(mu: MuTensor2D, sx, sy, sz):
    mu_xx_p = (sy*sz/sx) * mu.mu_xx
    mu_yy_p = (sz*sx/sy) * mu.mu_yy
    mu_zz_p = (sx*sy/sz) * mu.mu_zz

    return MuTensor2D(mu_xx_p, mu_yy_p, mu_zz_p)
