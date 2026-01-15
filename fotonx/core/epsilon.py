import numpy as np
from typing import Dict, Optional, Union
from fotonx.core.materials import MaterialDB

OverrideIso = Union[np.ndarray, Dict[int, complex]]

def eps_by_id_from_names(id_to_name: Dict[int, str],
                         mat_db: MaterialDB,
                         wvl_um: float) -> Dict[int, complex]:
    out: Dict[int, complex] = {}
    for mid, name in id_to_name.items():
        eps = mat_db.get(name).permittivity(wvl_um)
        out[int(mid)] = complex(eps)
    return out

def build_eps_iso(mat_id: np.ndarray,
                  eps_by_id: Dict[int, complex],
                  dtype=np.complex128) -> np.ndarray:
    """
    Return eps(x,y) complex map (Ny,Nx) using a LUT built from eps_by_id.
    """
    if not np.issubdtype(mat_id.dtype, np.integer):
        mat_id = mat_id.astype(np.int32, copy=False)

    # Safety: ensure every id used in mat_id exists in eps_by_id
    used_ids = set(np.unique(mat_id).tolist())
    known_ids = set(int(k) for k in eps_by_id.keys())
    missing = used_ids - known_ids
    if missing:
        raise ValueError(f"Missing eps values for material ids: {sorted(missing)}")

    max_id = int(mat_id.max())
    lut = np.zeros(max_id + 1, dtype=dtype)
    for mid, epsv in eps_by_id.items():
        lut[int(mid)] = complex(epsv)
    return lut[mat_id]

def build_eps_iso_with_override(mat_id: np.ndarray,
                                eps_by_id: Dict[int, complex],
                                override: Optional[OverrideIso] = None,
                                dtype=np.complex128) -> np.ndarray:
    """
    Build eps map then optionally override:
      - override = ndarray (Ny,Nx): replaces entire eps map
      - override = dict{mid: eps}: replaces eps where mat_id==mid
    """
    eps = build_eps_iso(mat_id, eps_by_id, dtype=dtype)

    if override is None:
        return eps

    if isinstance(override, np.ndarray):
        if override.shape != eps.shape:
            raise ValueError("override map must have same shape as mat_id")
        return override.astype(dtype, copy=False)

    if isinstance(override, dict):
        eps2 = eps.copy()
        for mid, val in override.items():
            eps2[mat_id == int(mid)] = complex(val)
        return eps2

    raise TypeError("override must be None, ndarray, or dict{mid: eps}")
