"""
Example 2 — Isotropic MMI / waveguide eigenmodes with a doped material (GDS + MaterialDB)
Adapted from: https://doi.org/10.1016/j.optcom.2006.10.084

This example targets an integrated photonic device (e.g., an MMI section)
where one region is modeled as an isotropic “doped” material using the MaterialDB.

The goal is to show an end-to-end workflow for:
- Reading multiple GDS cells (e.g., WG + MMI),
- Assigning different materials (including doped variants),
- Solving vectorial eigenmodes and plotting modal fields.

Features exercised in this script
---------------------------------
1) Multi-cell GDS workflow (device pieces)
   - Loads a GDS library and extracts polygons from multiple cells
     (e.g., "WG" and "MMI") using a shared layer_map.

2) Rasterization + per-cell grids
   - Builds grid(s) from polygons and rasterizes mat_id maps.
   - Demonstrates how to keep consistent dx, dy, padding across related cells.

3) MaterialDB extension: doped material
   - Creates a doped material model via Material.doped(...)
   - Adds it to the MaterialDB and maps specific polygon IDs to that doped material.

4) Isotropic epsilon -> tensor promotion
   - Builds ε(x,y) isotropic from eps_by_id and promotes it to diagonal tensor form.
   - Uses μ = 1 (diagonal), consistent with most non-magnetic photonic materials.

5) PML + full-vector eigenmode solve
   - Builds PML, applies it to ε and μ, assembles the system, solves eigenmodes.

6) Visualization for device sanity + modal inspection
   - Geometry plot for each cell (optional).
   - Field plots with polygon overlay to verify confinement and symmetry.

Data
----
GDS is expected in:
    examples/data/<mmi_or_device_file>.gds

Notes
-----
- Although the device is “MMI”, this example focuses on eigenmodes of a cross-section.
  (Propagation / BPM / EME is separate.)
- This example is intentionally isotropic to avoid tensor algebra complexity.
- For explicit anisotropic tensor entries (off-diagonals), see Example 3.

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
from fotonx.core.materials import MaterialDB, Material
from fotonx.solvers.fd_mode_vec.run import (
    GeometryCfg, GridCfg, PmlCfg, SolveCfg, MaterialCfg, VModeCfg, run_vmode
)

HERE = Path(__file__).resolve().parent          # .../examples
DATA = HERE / "data"                            # .../examples/data
gds_path = DATA / "MMI_test.gds"                # Path portable

# --- DOPPING MATERIAL ---
db = MaterialDB.default()
db.add("sio2_ge", Material.doped("sio2", delta=0.0075, new_name="sio2_ge"))

# --- CONFIG ---
cfg = VModeCfg(
    geom=GeometryCfg(
        gds_path=str(gds_path),
        cell_name="WG",  # eigenmodes of the WG cross-section (is possible to use "MMI" layer as well)
        layer_map={
            "si_layer": (1, 0),
            "sio2_layer": (2, 0),
            "sio2d_layer": (3, 0),
        },
        mat_ids={"air": 0, "sio2_layer": 1, "si_layer": 2, "sio2d_layer": 3},
        background_name="air",
    ),
    grid=GridCfg(
        dx=0.20,
        dy=0.20,
        pad_x=1.0,
        pad_y=1.0,
        wvl=1.55,
    ),
    pml=PmlCfg(
        thick_x_um=2.0,
        thick_y_um=2.0,
        alpha_x_max=3.0,
        alpha_y_max=3.0,
    ),
    solve=SolveCfg(
        num_modes=3,
        n_guess=1.45,
        z_fields=True,
    ),
    mats=MaterialCfg(
        # map *material IDs* -> DB material names
        id_to_db_map={
            0: "air",
            1: "sio2",
            2: "si",
            3: "sio2_ge",   # doped silica
        },
        eps_by_id=None,
        mu_by_id=None,
    ),
    provider="db_iso",
    provider_kwargs={"db": db},  # put the custom DB with doped material
)

print("SOLVING ...")

res, scene = run_vmode(cfg, return_scene=True)
polys = scene["polys"]
grid = scene["grid"]

print("neff =", res.neff)

# --- PLOTS ---

colors = {
    "si_layer":   "#1C5EB3",  # warm gold
    "sio2_layer": "#506368",  # light gray
    "sio2d_layer":"#1312127F",  # mid gray
}

cmap = mpl.colormaps["magma"].copy()
cmap.set_under("black")

# Visualize photonic device
plot_polys(polys, colors=colors)
plt.show()

# Access fields (example)
mode = 0
Ex = res.Ex[mode]
Ey = res.Ey[mode]
Ez = res.Ez[mode]

x, y = grid.x, grid.y
X, Y = np.meshgrid(x, y)   # x: (Nx,), y: (Ny,)

# normalize so that the major component has max = 1
Ex_max = np.max(np.abs(Ex))
Ey_max = np.max(np.abs(Ey))

# choose major component automatically
if Ex_max >= Ey_max:
    s = Ex_max
    major = "Ex"
else:
    s = Ey_max
    major = "Ey"

Exn = Ex / (s + 1e-30)
Eyn = Ey / (s + 1e-30)

# --- plot |Ex|, |Ey|, |Ez| as contours with polygon overlay
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

fields = [
    (r"|E$_x$|", np.abs(Ex)),
    (r"|E$_y$|", np.abs(Ey)),
    (r"|E$_z$|", np.abs(Ez)),
]

data = np.abs(Ex)  # or Ey/Ez
vmax = float(np.max(data) + 1e-30)
vmin = 1e-3 * vmax   # threshold for “black”

for ax, (ttl, data) in zip(axes, fields):
    levels = np.linspace(vmin, vmax, 15)
    cs = ax.contour(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
    fig.colorbar(cs, ax=ax)
    ax.set_facecolor("black")
    ax.set_title(f"{ttl}")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal", adjustable="box")

    overlay_polys(ax, polys, colors=colors, lw=2.0, alpha=1.0)

plt.show()


I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
plt.figure(figsize=(6,5))
plt.imshow(I, cmap=cmap, origin="lower", extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])), aspect="equal")
plt.colorbar(label="|E|^2")
ax = plt.gca()
overlay_polys(ax, polys, colors=colors, lw=2.0, alpha=1.0)
plt.title(f"|E|^2")
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.show()