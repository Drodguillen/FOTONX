from dataclasses import dataclass
import numpy as np

@dataclass
class EpsTensor2D:
    eps_xx: np.ndarray
    eps_yy: np.ndarray
    eps_zz: np.ndarray
    eps_xy: np.ndarray
    eps_yz: np.ndarray
    eps_xz: np.ndarray

@dataclass
class MuTensor2D:
    mu_xx: np.ndarray
    mu_yy: np.ndarray
    mu_zz: np.ndarray

@dataclass 
class VModeResult:
    beta: np.ndarray
    neff: np.ndarray
    Ex: list
    Ey: list
    Ez: list
    Hx: list
    Hy: list
    Hz: list

def eps_tensor_from_iso(
    eps_iso: np.ndarray,
    overrides: dict[str, np.ndarray] | None = None
) -> EpsTensor2D:
    """
    Build a full 2D permittivity tensor from an isotropic map.
    overrides can replace any component: keys in {"xx","yy","zz","xy","xz","yz"}.
    """
    if eps_iso.ndim != 2:
        raise ValueError("eps_iso must be (Ny,Nx)")

    Ny, Nx = eps_iso.shape
    z = np.zeros((Ny, Nx), dtype=eps_iso.dtype)

    exx = eps_iso.copy()
    eyy = eps_iso.copy()
    ezz = eps_iso.copy()
    exy = z.copy()
    exz = z.copy()
    eyz = z.copy()

    if overrides:
        for k, v in overrides.items():
            if v.shape != (Ny, Nx):
                raise ValueError(f"override[{k}] must have shape {(Ny,Nx)}")
            if k == "xx": exx = v
            elif k == "yy": eyy = v
            elif k == "zz": ezz = v
            elif k == "xy": exy = v
            elif k == "xz": exz = v
            elif k == "yz": eyz = v
            else:
                raise ValueError(f"Unknown override key '{k}'")

    return EpsTensor2D(
        eps_xx=exx, eps_yy=eyy, eps_zz=ezz,
        eps_xy=exy, eps_yz=eyz, eps_xz=exz
    )

def mu_tensor_from_iso(mu_iso, Ny=None, Nx=None, overrides=None) -> MuTensor2D:
    mu_iso = np.asarray(mu_iso)

    # allow scalar mu_iso (e.g., 1.0)
    if mu_iso.ndim == 0:
        if Ny is None or Nx is None:
            raise ValueError("Scalar mu_iso requires Ny,Nx")
        mu_iso = np.full((Ny, Nx), mu_iso, dtype=complex)

    if mu_iso.ndim != 2:
        raise ValueError("mu_iso must be scalar or (Ny,Nx)")

    Ny2, Nx2 = mu_iso.shape
    mxx = mu_iso.copy()
    myy = mu_iso.copy()
    mzz = mu_iso.copy()

    if overrides:
        for k, v in overrides.items():
            if v.shape != (Ny2, Nx2):
                raise ValueError(f"override[{k}] must have shape {(Ny2,Nx2)}")
            if k == "xx": mxx = v
            elif k == "yy": myy = v
            elif k == "zz": mzz = v
            else: raise ValueError(f"Unknown override key '{k}'")

    return MuTensor2D(mu_xx=mxx, mu_yy=myy, mu_zz=mzz)



