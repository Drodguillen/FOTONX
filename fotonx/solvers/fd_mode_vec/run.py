from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, overload, Literal

from fotonx.solvers.fd_mode_vec.builders import build_from_gds
from fotonx.solvers.fd_mode_vec.pml import build_pml, apply_pml_eps, apply_pml_mu, pml_um_to_px
from fotonx.solvers.fd_mode_vec.assemble import assemble_mts
from fotonx.solvers.fd_mode_vec.solve import solve_vec_modes
from fotonx.solvers.fd_mode_vec.providers import (
    TensorBuildCtx,
    DbIsoProvider,
    ConstTensorProvider,
    TensorFieldProvider,
)

from fotonx.solvers.fd_mode_vec.types import VModeResult  # <-- important


# -------------------- CFG --------------------

@dataclass
class GeometryCfg:
    gds_path: str
    cell_name: str
    layer_map: Dict[str, Tuple[int, int]]
    mat_ids: Dict[str, int]
    background_name: str = "air"


@dataclass
class GridCfg:
    dx: float
    dy: float
    pad_x: float
    pad_y: float
    wvl: float


@dataclass
class PmlCfg:
    thick_x_um: float
    thick_y_um: float
    alpha_x_max: float = 3.0
    alpha_y_max: float = 3.0


@dataclass
class SolveCfg:
    num_modes: int
    n_guess: float
    z_fields: bool = False


@dataclass
class MaterialCfg:
    id_to_db_map: Optional[Dict[int, str]] = None
    eps_by_id: Optional[Dict[int, Dict[str, Any]]] = None   # "xx","yy","zz","xy","xz","yz"
    mu_by_id: Optional[Dict[int, Dict[str, Any]]] = None    # "xx","yy","zz"
    db_name: str = "default"


@dataclass
class VModeCfg:
    geom: GeometryCfg
    grid: GridCfg
    pml: PmlCfg
    solve: SolveCfg
    mats: MaterialCfg
    provider: str = "db_iso"          # "db_iso" | "const_tensor" | "tensor_field"
    provider_kwargs: Optional[Dict[str, Any]] = None


# -------------------- TYPES --------------------

SceneDict = Dict[str, Any]
RunVModeReturn = Union[VModeResult, Tuple[VModeResult, SceneDict]]


# -------------------- PROVIDER --------------------

def _make_provider(cfg: VModeCfg):
    if cfg.provider == "db_iso":
        if cfg.mats.id_to_db_map is None:
            raise ValueError("mats.id_to_db_map is required for provider='db_iso'")

        db = None
        if cfg.provider_kwargs is not None:
            db = cfg.provider_kwargs.get("db", None)

        return DbIsoProvider(
            id_to_dbname=cfg.mats.id_to_db_map,
            eps_by_id=cfg.mats.eps_by_id,
            mu_by_id=cfg.mats.mu_by_id,
            db=db,
        )

    if cfg.provider == "const_tensor":
        if cfg.mats.eps_by_id is None:
            raise ValueError("mats.eps_by_id is required for provider='const_tensor'")
        base_eps = 1.0
        if cfg.provider_kwargs is not None:
            base_eps = float(cfg.provider_kwargs.get("base_eps_iso", base_eps))
        return ConstTensorProvider(
            eps_by_id=cfg.mats.eps_by_id,
            mu_by_id=cfg.mats.mu_by_id,
            base_eps_iso=base_eps,
        )

    if cfg.provider == "tensor_field":
        if not cfg.provider_kwargs or "tensor_builder" not in cfg.provider_kwargs:
            raise ValueError("provider_kwargs['tensor_builder'] is required for provider='tensor_field'")
        return TensorFieldProvider(cfg.provider_kwargs["tensor_builder"])

    raise ValueError(f"Unknown provider '{cfg.provider}'")


# -------------------- OVERLOADS --------------------

@overload
def run_vmode(cfg: VModeCfg, return_scene: Literal[False] = False) -> VModeResult: ...
@overload
def run_vmode(cfg: VModeCfg, return_scene: Literal[True]) -> Tuple[VModeResult, SceneDict]: ...


def run_vmode(cfg: VModeCfg, return_scene: bool = False) -> RunVModeReturn:
    # 1) Geometry -> polys, grid, mat_id
    polys, grid, mat_id = build_from_gds(cfg.geom, cfg.grid)
    Nx, Ny = grid.Nx, grid.Ny

    # 2) Build ctx for provider
    ctx = TensorBuildCtx(
        wvl=cfg.grid.wvl,
        dx=cfg.grid.dx,
        dy=cfg.grid.dy,
        Nx=Nx,
        Ny=Ny,
        x=grid.x,
        y=grid.y,
        mat_id=mat_id,
        db=None,
        eps_iso=None,
    )

    # 3) Build tensors
    provider = _make_provider(cfg)
    eps_t, mu_t = provider.build(ctx)

    # 4) PML
    npml_x, npml_y = pml_um_to_px(cfg.pml.thick_x_um, cfg.pml.thick_y_um, cfg.grid.dx, cfg.grid.dy)
    sx, sy, sz = build_pml(
        Nx, Ny, cfg.grid.dx, cfg.grid.dy,
        npml_x=npml_x,
        npml_y=npml_y,
        alpha_x_max=cfg.pml.alpha_x_max,
        alpha_y_max=cfg.pml.alpha_y_max,
    )
    eps_pml = apply_pml_eps(eps_t, sx, sy, sz)
    mu_pml = apply_pml_mu(mu_t, sx, sy, sz)

    # 5) Assemble
    A = assemble_mts(eps_pml, mu_pml, Nx, Ny, cfg.grid.dx, cfg.grid.dy, cfg.grid.wvl)

    # 6) Solve
    res = solve_vec_modes(
        A, cfg.grid.wvl, Ny, Nx,
        cfg.solve.num_modes,
        cfg.solve.n_guess,
        z_fields=cfg.solve.z_fields,
        eps_used=eps_pml if cfg.solve.z_fields else None,
        mu_used=mu_pml if cfg.solve.z_fields else None,
        dx=cfg.grid.dx,
        dy=cfg.grid.dy,
    )

    if return_scene:
        scene: SceneDict = {
            "polys": polys,
            "grid": grid,
            "mat_id": mat_id,
            "ctx": ctx,
            "eps_t": eps_t,
            "mu_t": mu_t,
            "eps_pml": eps_pml,
            "mu_pml": mu_pml,
            "A": A,
        }
        return res, scene

    return res


