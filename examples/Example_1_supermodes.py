"""
Example 1 — Vectorial FDFD eigenmodes from a GDS cross-section (isotropic materials)

This example demonstrates the end-to-end workflow of the FOTONX vmode solver
for a simple integrated waveguide geometry defined in GDS (e.g., Si3N4 core on SiO2).

Features exercised in this script
---------------------------------
1) GDS import + polygon extraction
   - Reads a .gds file and selects a target cell (e.g., "WG").
   - Uses a layer_map to assign physical layers to polygons.

2) Material assignment (isotropic, database-driven)
   - Maps polygon IDs -> material names using MaterialDB (provider="db_iso").
   - Builds an isotropic ε(x,y) and μ(x,y) and internally promotes them to diagonal tensors.

3) Grid generation + padding
   - Creates a uniform grid using (dx, dy) in microns.
   - Adds padding (pad_x, pad_y) around the imported geometry.

4) PML boundary conditions
   - Builds a separable PML along x and y with configurable thickness and strength.

5) Full-vector eigenmode solve (Ex, Ey, Ez, Hx, Hy, Hz)
   - Solves for the requested number of guided modes using an effective index guess (n_guess).
   - Reconstructs longitudinal fields when z_fields=True.

6) Visualization utilities
   - Plots the GDS polygons for sanity check.
   - Plots |Ex|, |Ey| and Re(Ex), Re(Ey) with polygon overlays.
   - Applies a simple phase-fix so real-part plots are physically meaningful.

Data
----
The required GDS file is stored in:
    examples/data/<your_gds_file>.gds

Notes
-----
- This example is intentionally isotropic to keep the pipeline minimal.
- For anisotropic / manual tensor overrides, see Example 3.
- For fiber-like geometries and window cropping, see Example 4.

Author / Autoría
----------------
Daniel Rodríguez Guillén (c) 2025–2026

License / Licencia
------------------
Non-commercial academic/research use only. Attribution required.
See repository LICENSE for the full terms.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

from fotonx.core.plots import plot_polys, overlay_polys
from fotonx.solvers.fd_mode_vec.run import (
    GeometryCfg, GridCfg, PmlCfg, SolveCfg, MaterialCfg, VModeCfg, run_vmode
)

HERE = Path(__file__).resolve().parent          # .../examples
DATA = HERE / "data"                            # .../examples/data
gds_path = DATA / "supermode_test.gds"                # Path portable

layer_map = {
    "sio2_layer":  (1, 0),
    "si3n4_layer": (2, 0),
}
mat_ids = {"air": 0, "sio2_layer": 1, "si3n4_layer": 2}

# DB names for each ID
id_to_db = {0: "air", 1: "sio2", 2: "si3n4"}

# --- CONFIG ---
cfg = VModeCfg(
    geom=GeometryCfg(
        gds_path=str(gds_path),
        cell_name="WG",
        layer_map=layer_map,
        mat_ids=mat_ids,
        background_name="air",
    ),
    grid=GridCfg(
        dx=0.04,
        dy=0.04,
        pad_x=1.0,
        pad_y=2.0,
        wvl=1.55,     # um
    ),
    pml=PmlCfg(
        thick_x_um=1.0,
        thick_y_um=1.0,
        alpha_x_max=3.0,
        alpha_y_max=3.0,
    ),
    solve=SolveCfg(
        num_modes=4,
        # good guess: ~n_core
        n_guess=1.8,
        z_fields=True,      # Ez and Hz
    ),
    mats=MaterialCfg(
        id_to_db_map=id_to_db,
        eps_by_id=None,     # optional overrides
        mu_by_id=None,      # optional overrides
    ),
    provider="db_iso",      # database isotropic
)

print("SOLVING ...")

res, scene = run_vmode(cfg, return_scene=True)
polys = scene["polys"]
grid = scene["grid"]

print("neff =", res.neff)

# --- PLOTS ---

colors = {
    "sio2_layer":   "#2E2C29",  # warm gold
    "si3n4_layer": "#4179A1",  # light gray
}

cmap = mpl.colormaps["magma"].copy()
cmap.set_under("black")

# Visualize photonic device
plot_polys(polys, colors=colors)
plt.show()


x, y = grid.x, grid.y
X, Y = np.meshgrid(x, y)   # x: (Nx,), y: (Ny,)
mode = 0

Ex = res.Ex[mode]
Ey = res.Ey[mode]
Ez = res.Ez[mode]

Hx = res.Hx[mode]
Hy = res.Hy[mode]
Hz = res.Hz[mode]

# Phase-fix using Ey so Re() is meaningful
iy, ix = np.unravel_index(np.argmax(np.abs(Ey)), Ey.shape)
ph = np.exp(-1j*np.angle(Ey[iy, ix]))
Ex = Ex * ph
Ey = Ey * ph

# Normalized intensities (0..1) using global max of transverse field
Et = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
scaleI = float(Et.max() + 1e-30)

Ix = np.abs(Ex) / scaleI
Iy = np.abs(Ey) / scaleI

# Signed real parts normalized to [-1..1] per component
Rx = np.real(Ex)
Ry = np.real(Ey)
Rx = Rx / (np.max(np.abs(Rx)) + 1e-30)
Ry = Ry / (np.max(np.abs(Ry)) + 1e-30)

# Colormaps
cmapI = mpl.colormaps["magma"].copy()
cmapI.set_under("black")
cmapS = mpl.colormaps["RdBu_r"]

# Figure 2x2 
fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)

# --- |Ex|
im = ax[0,0].imshow(Ix, origin="lower",
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    cmap=cmapI, vmin=1e-3, vmax=1.0)
ax[0,0].set_title(r"$|E_x|$")
ax[0,0].set_xlabel("x (um)"); ax[0,0].set_ylabel("y (um)")
ax[0,0].set_aspect("equal", adjustable="box")
ax[0,0].set_facecolor("black")
overlay_polys(ax[0,0], polys, colors=colors, lw=2.0, alpha=1.0)
fig.colorbar(im, ax=ax[0,0], fraction=0.046, pad=0.04)

# --- |Ey|
im = ax[0,1].imshow(Iy, origin="lower",
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    cmap=cmapI, vmin=1e-3, vmax=1.0)
ax[0,1].set_title(r"$|E_y|$")
ax[0,1].set_xlabel("x (um)"); ax[0,1].set_ylabel("y (um)")
ax[0,1].set_aspect("equal", adjustable="box")
ax[0,1].set_facecolor("black")
overlay_polys(ax[0,1], polys, colors=colors, lw=2.0, alpha=1.0)
fig.colorbar(im, ax=ax[0,1], fraction=0.046, pad=0.04)

# --- Re(Ex)
im = ax[1,0].imshow(Rx, origin="lower",
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    cmap=cmapS, vmin=-1.0, vmax=1.0)
ax[1,0].set_title(r"$\mathrm{Re}(E_x)$ / max")
ax[1,0].set_xlabel("x (um)"); ax[1,0].set_ylabel("y (um)")
ax[1,0].set_aspect("equal", adjustable="box")
overlay_polys(ax[1,0], polys, colors=colors, lw=2.0, alpha=1.0)
fig.colorbar(im, ax=ax[1,0], fraction=0.046, pad=0.04)

# --- Re(Ey)
im = ax[1,1].imshow(Ry, origin="lower",
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    cmap=cmapS, vmin=-1.0, vmax=1.0)
ax[1,1].set_title(r"$\mathrm{Re}(E_y)$ / max")
ax[1,1].set_xlabel("x (um)"); ax[1,1].set_ylabel("y (um)")
ax[1,1].set_aspect("equal", adjustable="box")
overlay_polys(ax[1,1], polys, colors=colors, lw=2.0, alpha=1.0)
fig.colorbar(im, ax=ax[1,1], fraction=0.046, pad=0.04)

plt.show()