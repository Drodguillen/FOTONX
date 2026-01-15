from fotonx.core.materials import MaterialDB
from fotonx.core.epsilon import eps_by_id_from_names, build_eps_iso
from fotonx.solvers.fd_mode_vec.types import eps_tensor_from_iso, mu_tensor_from_iso
import numpy as np

def build_component_override_arrays(mat_id, Ny, Nx, by_id_dict, valid_keys):
    # returns dict component->(Ny,Nx) arrays, only for keys provided
    out = {}
    if not by_id_dict:
        return out

    for mid, comps in by_id_dict.items():
        mask = (mat_id == mid)
        for k, v in comps.items():
            if k not in valid_keys:
                raise ValueError(f"Invalid component '{k}'. Valid: {valid_keys}")
            if k not in out:
                out[k] = np.zeros((Ny, Nx), dtype=complex)
                out[k][:] = np.nan  # mark as "unset"
            if np.isscalar(v):
                out[k][mask] = complex(v) #ignore
            else:
                arr = np.asarray(v)
                if arr.shape != (Ny, Nx):
                    raise ValueError(f"Override {k} for id {mid} must be (Ny,Nx)")
                out[k][mask] = arr[mask]

    # Replace nan by "no override"
    for k in list(out.keys()):
        if np.all(np.isnan(out[k])):
            del out[k]
        else:
            out[k] = np.where(np.isnan(out[k]), 0.0, out[k])
    return out

# fotonx/solvers/fd_mode_vec/providers.py
from dataclasses import dataclass
import numpy as np

@dataclass
class TensorBuildCtx:
    wvl: float
    dx: float
    dy: float
    Nx: int
    Ny: int
    x: np.ndarray
    y: np.ndarray
    mat_id: np.ndarray
    db: object | None = None
    eps_iso: np.ndarray | None = None 


class DbIsoProvider:
    def __init__(self, id_to_dbname, eps_by_id=None, mu_by_id=None, db=None):
        self.id_to_dbname = id_to_dbname
        self.eps_by_id = eps_by_id
        self.mu_by_id  = mu_by_id
        self.db = db

    def build(self, ctx):
        Ny, Nx = ctx.Ny, ctx.Nx
        db = self.db if self.db is not None else MaterialDB.default()

        eps_lookup = eps_by_id_from_names(self.id_to_dbname, db, wvl_um=ctx.wvl)
        eps_iso = build_eps_iso(ctx.mat_id, eps_lookup)   # (Ny,Nx)
        ctx.eps_iso = eps_iso

        eps_over = build_component_override_arrays(ctx.mat_id, Ny, Nx, self.eps_by_id,
                                                   valid_keys={"xx","yy","zz","xy","xz","yz"})
        eps_t = eps_tensor_from_iso(eps_iso, overrides=eps_over if eps_over else None)

        mu_t = mu_tensor_from_iso(1.0, Ny, Nx)
        mu_over = build_component_override_arrays(ctx.mat_id, Ny, Nx, self.mu_by_id,
                                                  valid_keys={"xx","yy","zz"})
        if mu_over:
            # apply mu overrides directly
            if "xx" in mu_over: mu_t.mu_xx = mu_over["xx"]
            if "yy" in mu_over: mu_t.mu_yy = mu_over["yy"]
            if "zz" in mu_over: mu_t.mu_zz = mu_over["zz"]

        return eps_t, mu_t

class ConstTensorProvider:
    def __init__(self, eps_by_id, mu_by_id=None, base_eps_iso=1.0):
        self.eps_by_id = eps_by_id
        self.mu_by_id  = mu_by_id
        self.base_eps_iso = base_eps_iso  # background default (air)

    def build(self, ctx):
        Ny, Nx = ctx.Ny, ctx.Nx

        # base isotropic map
        eps_iso = np.full((Ny, Nx), complex(self.base_eps_iso), dtype=complex)

        eps_over = build_component_override_arrays(ctx.mat_id, Ny, Nx, self.eps_by_id,
                                                   valid_keys={"xx","yy","zz","xy","xz","yz"})
        eps_t = eps_tensor_from_iso(eps_iso, overrides=eps_over)

        mu_t = mu_tensor_from_iso(1.0, Ny, Nx)
        mu_over = build_component_override_arrays(ctx.mat_id, Ny, Nx, self.mu_by_id,
                                                  valid_keys={"xx","yy","zz"})
        if mu_over:
            if "xx" in mu_over: mu_t.mu_xx = mu_over["xx"]
            if "yy" in mu_over: mu_t.mu_yy = mu_over["yy"]
            if "zz" in mu_over: mu_t.mu_zz = mu_over["zz"]
        return eps_t, mu_t

class TensorFieldProvider:
    def __init__(self, tensor_builder_callable):
        self.tensor_builder = tensor_builder_callable

    def build(self, ctx):
        eps_t, mu_t = self.tensor_builder(ctx)
        return eps_t, mu_t
