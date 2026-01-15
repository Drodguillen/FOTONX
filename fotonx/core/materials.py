from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Dict, Tuple, Union

import numpy as np

from fotonx.core.materials_db import MATERIAL_TABLE


ArrayLike = Union[float, np.ndarray]


@dataclass(frozen=True)
class Material:
    """
    Dispersive material model:
      eps(λ) = 1 + Σ_i [ B_i * λ^2 / (λ^2 - A_i) ]      (λ in µm, A in µm^2)
      then apply optional doping factor delta on sqrt(eps): eps *= (1+delta)^2

    If epsConst is provided (not NaN), eps is constant (no Sellmeier).
    If muConst is provided (not NaN), mu is constant; else mu=1.
    """
    name: str
    A: np.ndarray          # (M,) in um^2
    B: np.ndarray          # (M,)
    n_nl: float = 0.0
    delta: float = 0.0     # fractional Δn applied on sqrt(eps)
    epsConst: float = np.nan
    muConst: float = np.nan

    def permittivity(self, wvl_um: ArrayLike) -> np.ndarray:
        wvl = np.asarray(wvl_um, dtype=float)
        if np.any(wvl <= 0):
            raise ValueError("wvl_um must be positive.")

        # Constant eps if provided
        if not np.isnan(self.epsConst):
            base = np.full_like(wvl, float(self.epsConst), dtype=float)
            return base * (1.0 + float(self.delta)) ** 2

        # No Sellmeier coeffs -> vacuum eps=1
        if self.A.size == 0:
            base = np.ones_like(wvl, dtype=float)
            return base * (1.0 + float(self.delta)) ** 2

        if self.A.shape != self.B.shape:
            raise ValueError(f"A and B must have same shape. Got {self.A.shape} and {self.B.shape}")

        lam2 = wvl * wvl  # (N,)
        A = self.A.reshape(-1, 1)     # (M,1)
        B = self.B.reshape(-1, 1)     # (M,1)
        lam2r = lam2.reshape(1, -1)   # (1,N)

        # Avoid division warnings at resonances; user should avoid exact poles anyway.
        term = (B * lam2r) / (lam2r - A)  # (M,N)
        base = 1.0 + np.sum(term, axis=0) # (N,)

        return base * (1.0 + float(self.delta)) ** 2

    def permeability(self, wvl_um: ArrayLike) -> np.ndarray:
        wvl = np.asarray(wvl_um, dtype=float)
        if np.any(wvl <= 0):
            raise ValueError("wvl_um must be positive.")

        if not np.isnan(self.muConst):
            return np.full_like(wvl, float(self.muConst), dtype=float)

        return np.ones_like(wvl, dtype=float)

    @staticmethod
    def from_string(name: str) -> "Material":
        key = name.lower()
        if key not in MATERIAL_TABLE:
            raise ValueError(f"Unknown material '{name}'")

        entry = MATERIAL_TABLE[key]
        A = np.asarray(entry.get("A", np.array([], dtype=float)), dtype=float).ravel()
        B = np.asarray(entry.get("B", np.array([], dtype=float)), dtype=float).ravel()
        n_nl = float(entry.get("n_nl", 0.0))

        return Material(name=key, A=A.copy(), B=B.copy(), n_nl=n_nl)

    @staticmethod
    def constant(eps0: float, mu0: float = 1.0, name: str = "constant") -> "Material":
        eps0 = float(eps0)
        mu0 = float(mu0)
        if eps0 <= 0 or mu0 <= 0:
            raise ValueError("eps0 and mu0 must be positive.")
        return Material(
            name=name,
            A=np.array([], dtype=float),
            B=np.array([], dtype=float),
            n_nl=0.0,
            delta=0.0,
            epsConst=eps0,
            muConst=mu0,
        )

    @staticmethod
    def doped(
        base: Union["Material", str, float, int, Tuple[float, float], np.ndarray],
        delta: float,
        new_name: str = "",
    ) -> "Material":
        """
        Returns a new Material that shares the same dispersion as `base` but
        applies a constant refractive index increase:
            n -> n*(1+delta)  ==>  eps -> eps*(1+delta)^2

        base can be:
          - Material
          - str (lookup in MATERIAL_TABLE)
          - float/int (eps0, mu=1)
          - (eps0, mu0)
          - np.ndarray with 1 or 2 elements (eps0 or (eps0, mu0))
        """
        delta = float(delta)
        if delta < 0:
            raise ValueError("delta must be >= 0")

        host = _to_material(base)

        if new_name == "":
            new_name = f"{host.name}_d{delta:g}"

        return Material(
            name=new_name,
            A=host.A.copy(),
            B=host.B.copy(),
            n_nl=host.n_nl,
            delta=delta,
            epsConst=float(host.epsConst),
            muConst=float(host.muConst),
        )


@singledispatch
def _to_material(base) -> Material:
    raise TypeError("base must be Material, str, numeric, (eps0,mu0), or np.ndarray.")


@_to_material.register
def _(base: Material) -> Material:
    return base


@_to_material.register
def _(base: str) -> Material:
    return Material.from_string(base)


@_to_material.register
def _(base: int) -> Material:
    return Material.constant(float(base), 1.0)


@_to_material.register
def _(base: float) -> Material:
    return Material.constant(float(base), 1.0)


@_to_material.register
def _(base: tuple) -> Material:
    if len(base) != 2:
        raise ValueError("tuple base must be (eps0, mu0).")
    return Material.constant(float(base[0]), float(base[1]))


@_to_material.register
def _(base: np.ndarray) -> Material:
    arr = np.asarray(base).ravel()
    if arr.size == 1:
        return Material.constant(float(arr[0]), 1.0)
    if arr.size == 2:
        return Material.constant(float(arr[0]), float(arr[1]))
    raise ValueError("np.ndarray base must contain eps0 or (eps0, mu0).")


class MaterialDB:
    """
    Small helper: stores Material objects by name.
    """
    def __init__(self) -> None:
        self._db: Dict[str, Material] = {}

    def add(self, name: str, mat: Material) -> None:
        self._db[name.lower()] = mat

    def get(self, name: str) -> Material:
        key = name.lower()
        if key not in self._db:
            raise KeyError(f"Material '{name}' not in DB")
        return self._db[key]

    @staticmethod
    def default() -> "MaterialDB":
        db = MaterialDB()
        for k in MATERIAL_TABLE.keys():
            db.add(k, Material.from_string(k))
        return db


