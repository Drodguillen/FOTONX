"""
Example 4 — Photonic crystal fiber (PCF) cross-section with window cropping (foundation for Transformation Optics)

This example shows how to work with fiber-like geometries (PCF) where the full
transversal cross-section can be too large to simulate at high resolution.
Instead, we crop the simulation window to include only a limited number of air-hole rings.

The objective is to provide a clean, reproducible pipeline for:
- GDS-based PCF geometry,
- Custom simulation window,
- Rasterization and vectorial eigenmodes,
- A future extension point for transformation optics tensors.

Features exercised in this script
---------------------------------
1) GDS import + PCF polygon extraction
   - Extracts air-hole polygons from a PCF cell using a layer_map.

2) Window-controlled grid (cropping)
   - Uses a fixed window [xmin,xmax]×[ymin,ymax] (make_grid_from_window)
     instead of bounding-box padding around the whole structure.
   - This is essential when simulating only a few rings of holes.

3) Rasterization into mat_id
   - Rasterizes the cropped window with a background material (silica) and holes (air).

4) Isotropic assignment (baseline)
   - Builds isotropic ε(x,y) and μ(x,y) (non-magnetic) as a baseline.
   - Solves guided modes and plots modal fields.

5) Extension point: spatially varying tensors (twist / TO)
   - This example is designed so that ε and μ can later be replaced by pixel-wise
     tensor fields (provider="tensor_field") for permittivity transformations.
   - Recommended approach: build ε_tensor(x,y) from a Jacobian-based transform.

6) Visualization
   - Geometry plot + field contours with polygon overlays for sanity checks.

Data
----
GDS is expected in:
    examples/data/<pcf_test_file>.gds

Notes
-----
- Cropping changes boundary proximity: PML thickness and window size matter.
- The “permittivity transform” extension is expected to introduce off-diagonal coupling (ε terms),
  and potentially effective μ terms depending on the chosen TO formulation.

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

from fotonx.core.gds import read_gds_lib, get_cell, list_layers, extract_polygons
from fotonx.core.plots import plot_polys
from fotonx.core.raster import make_grid_from_polys, rasterize_mat_id, make_grid_from_window
from fotonx.core.materials import MaterialDB, Material
from fotonx.core.epsilon import eps_by_id_from_names, build_eps_iso
from fotonx.solvers.fd_mode_vec.types import eps_tensor_from_iso, mu_tensor_from_iso
from fotonx.solvers.fd_mode_vec.pml import build_pml, apply_pml_eps, apply_pml_mu
from fotonx.core.plots import overlay_polys
from fotonx.solvers.fd_mode_vec.assemble import assemble_mts
from fotonx.solvers.fd_mode_vec.solve import solve_vec_modes

from fotonx.solvers.fd_mode_vec.run import (
    GeometryCfg, GridCfg, PmlCfg, SolveCfg, MaterialCfg, VModeCfg
)
from fotonx.core.plots import overlay_polys

# READ GDS
HERE = Path(__file__).resolve().parent          # .../examples
DATA = HERE / "data"                            # .../examples/data
gds_path = DATA / "anisotropy_test.gds"                # Path portable

lib = read_gds_lib(str(gds_path))
pcf_cell = get_cell(lib, "PCF")

pcf_layers = list_layers(pcf_cell)

layer_map = {
        "air_layer": (2,0),
        }

pcf_polys = extract_polygons(pcf_cell, layer_map, False)

colors = {
    "air_layer": "#E6E6E6",  # light gray
}

cmap = mpl.colormaps["magma"].copy()
cmap.set_under("black")

plot_polys(pcf_polys, colors=colors)
plt.show(block=False)

plt.show()

# Assign an Id to each material
mat_ids = {"sio2":0, "air_layer": 1}

# Make grid
wvl = 1.55
omega = 2*np.pi/wvl
dx = 0.1
dy = 0.1
pad_x = 0.0
pad_y = 0.0
#grid_pcf = make_grid_from_polys(pcf_polys, dx=dx, dy=dy, pad_x=pad_x, pad_y=pad_y)
grid_pcf =  make_grid_from_window(-7, 7, -7, 7, dx, dy)
mat_id_pcf = rasterize_mat_id(grid_pcf, pcf_polys, mat_ids, background_name="sio2")

# Map the ID to the material name
id_to_name = {0:"sio2", 1:"air"}

Nx, Ny = grid_pcf.Nx, grid_pcf.Ny

# Mapping id to the material name
# Extract dabatase
db = MaterialDB.default()

# Build the permittivity map
eps_by_id = eps_by_id_from_names(id_to_name, db, wvl_um=wvl)
eps_map   = build_eps_iso(mat_id_pcf, eps_by_id)

# Build the epsilon tensor
eps_t = eps_tensor_from_iso(eps_map)
mu_t = mu_tensor_from_iso(1.0, Ny, Nx)

pml_thick_x = 2.0
pml_thick_y = 2.0

npml_x = int(round(pml_thick_x / dx))
npml_y = int(round(pml_thick_y / dy))

sx, sy, sz = build_pml(Nx, Ny, dx, dy, npml_x=npml_x, npml_y=npml_y, alpha_x_max=3.0, alpha_y_max=3.0)
eps_pml = apply_pml_eps(eps_t, sx, sy, sz)
mu_pml = apply_pml_mu(mu_t, sx, sy, sz)

A = assemble_mts(eps_pml, mu_pml, Nx, Ny, dx, dy, wvl)

# solving eigenvalues
num_modes = 10
#core_id = mat_ids["sio2"]
#n_core = np.sqrt(np.real(eps_by_id[core_id]))  # eps_by_id[mid] = complex eps
n_core = 1.41
print(" SOLVING ... ")
ModeResults = solve_vec_modes(A, wvl, Ny, Nx, num_modes, n_core, z_fields=True, eps_used=eps_pml, mu_used=mu_pml, dx=dx, dy=dy)

neff = ModeResults.neff
print("neff = ", neff)

x, y = grid_pcf.x, grid_pcf.y
X, Y = np.meshgrid(x, y)   # x: (Nx,), y: (Ny,)
mode = 1

Ex = ModeResults.Ex[mode]
Ey = ModeResults.Ey[mode]
Ez = ModeResults.Ez[mode]

Hx = ModeResults.Hx[mode]
Hy = ModeResults.Hy[mode]
Hz = ModeResults.Hz[mode]

# normalize so that the *major* component has max = 1
Ex_max = np.max(np.abs(Ex))
Ey_max = np.max(np.abs(Ey))

# choose major component automatically (like the paper’s statement)
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

    overlay_polys(ax, pcf_polys, colors=colors, lw=2.0, alpha=1.0)

plt.show()

I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
plt.figure(figsize=(6,5))
plt.imshow(I, cmap=cmap, origin="lower", extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])), aspect="equal")
plt.colorbar(label="|E|^2")
ax = plt.gca()
overlay_polys(ax, pcf_polys, colors=colors, lw=2.0, alpha=1.0)
plt.title(f"|E|^2")
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.show()

