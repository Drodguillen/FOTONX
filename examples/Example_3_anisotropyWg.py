"""
Example 3 — Manual anisotropic tensor override (const_tensor) + parameter sweep (paper-style)

This example reproduces an anisotropic waveguide scenario from the literature,
where the permittivity tensor ε is explicitly constructed from director angles
Adapted from: https://doi.org/10.1364/OE.17.005965

The key point: we bypass MaterialDB isotropic assignment and instead provide
full tensor components (including off-diagonals) via provider="const_tensor".

Features exercised in this script
---------------------------------
1) GDS import + polygon extraction
   - Reads a .gds file and extracts multiple layers (e.g., glass + LC core).

2) Manual tensor construction (per material ID)
   - Builds ε_xx, ε_yy, ε_zz and off-diagonals ε_xy, ε_xz, ε_yz from angles.
   - Demonstrates how to enforce non-zero ε_zz everywhere to avoid singular assembly.

3) Provider: const_tensor (override-driven)
   - Assigns tensor components per material ID (air, substrate, anisotropic core).
   - This is the intended workflow for analytic anisotropy and transformation optics prototypes.

4) Parameter sweeps (phi sweep + multiple n_guess)
   - Sweeps phi (director azimuth) and optionally tests multiple n_guess values
     to check eigen-solver robustness and mode tracking.

5) PML + full-vector solve + z-field reconstruction
   - Applies PML consistently to ε and μ and solves for guided modes.
   - Reconstructs Ez and Hz when z_fields=True.

6) Paper-style visualization
   - Produces |Ex| and |Ey| contour-style plots with polygon overlays (as in the paper).
   - Provides neff trends vs angle (phi) for multiple modes.

Data
----
GDS is expected in:
    examples/data/<anisotropy_test_file>.gds

Notes
-----
- This example is the reference “advanced anisotropy” template for FOTONX.
- For spatially varying tensors (pixel-wise ε tensor fields), see Example 4.

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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

from fotonx.solvers.fd_mode_vec.run import (
    GeometryCfg, GridCfg, PmlCfg, SolveCfg, MaterialCfg, VModeCfg, run_vmode
)
from fotonx.core.plots import overlay_polys


# ---- CONFIG ----

HERE = Path(__file__).resolve().parent          # .../examples
DATA = HERE / "data"                            # .../examples/data
gds_path = DATA / "anisotropy_test.gds"                # Path portable

cfg = VModeCfg(
    geom=GeometryCfg(
        gds_path=str(gds_path),
        cell_name="TOP",
        layer_map={
            "glass": (1, 0),
            "lc_core": (2, 0),
        },
        mat_ids={"air": 0, "glass": 1, "lc_core": 2},
        background_name="air",
    ),
    grid=GridCfg(
        dx=0.20,
        dy=0.20,
        pad_x=2.0,
        pad_y=2.0,
        wvl=1.55,
    ),
    pml=PmlCfg(
        thick_x_um=2.0,
        thick_y_um=2.0,
        alpha_x_max=3.0,
        alpha_y_max=3.0,
    ),
    solve=SolveCfg(
        num_modes=4,
        n_guess=1.50,   # overwritten inside the guess loop
        z_fields=True,
    ),
    mats=MaterialCfg(
        id_to_db_map=None,
        eps_by_id=None,   # built inside phi loop
        mu_by_id=None,
        db_name="default",
    ),
    provider="const_tensor",
    provider_kwargs={
        "base_eps_iso": (1.45**2)
    },
)

theta_c_deg = 30.0
phi_list_deg = [0.0, 30.0, 60.0, 90.0]

n_g = 1.45
eps_glass = n_g**2

n_o = 1.5292
n_e = 1.7072
eps_o = n_o**2
eps_e = n_e**2
Delta = (eps_e - eps_o)

theta = np.deg2rad(theta_c_deg)

n_guess_list = [1.55, 1.52, 1.515, 1.50]

colors = {
    "glass":   "#506368",
    "lc_core": "#1C5EB3",
    "air":     "#1312127F",
}
cmap = mpl.colormaps["magma"].copy()
cmap.set_under("black")

P = len(phi_list_deg)
M = cfg.solve.num_modes

neff_sel = np.full((P, M), np.nan + 1j*np.nan, dtype=complex)
Ex_sel   = [[None]*M for _ in range(P)]
Ey_sel   = [[None]*M for _ in range(P)]
scene_sel = [None]*P

mid_air   = cfg.geom.mat_ids["air"]
mid_glass = cfg.geom.mat_ids["glass"]
mid_lc    = cfg.geom.mat_ids["lc_core"]


for ip, phi_c_deg in enumerate(phi_list_deg):
    phi = np.deg2rad(phi_c_deg)

    ux = np.sin(theta) * np.cos(phi)
    uy = np.sin(theta) * np.sin(phi)
    uz = np.cos(theta)

    eps_xx_core = eps_o + Delta * (ux * ux)
    eps_yy_core = eps_o + Delta * (uy * uy)
    eps_zz_core = eps_o + Delta * (uz * uz)

    eps_xy_core = Delta * (ux * uy)
    eps_xz_core = Delta * (ux * uz)
    eps_yz_core = Delta * (uy * uz)

    cfg.mats.eps_by_id = {
        mid_air: {
            "xx": 1.0, "yy": 1.0, "zz": 1.0,
            "xy": 0.0, "xz": 0.0, "yz": 0.0,
        },
        mid_glass: {
            "xx": eps_glass, "yy": eps_glass, "zz": eps_glass,
            "xy": 0.0, "xz": 0.0, "yz": 0.0,
        },
        mid_lc: {
            "xx": eps_xx_core, "yy": eps_yy_core, "zz": eps_zz_core,
            "xy": eps_xy_core, "xz": eps_xz_core, "yz": eps_yz_core,
        },
    }

    ok = False
    last_err = None

    for n_guess in n_guess_list:
        cfg.solve.n_guess = float(n_guess)
        try:
            res, scene = run_vmode(cfg, return_scene=True)
            ok = True
            break
        except Exception as e:
            last_err = e
            ok = False

    if not ok:
        raise RuntimeError(f"All guesses failed for phi_c={phi_c_deg}°. Last error: {last_err}")

    print(f"phi_c={phi_c_deg:5.1f}°  used n_guess={cfg.solve.n_guess:.3f}  neff={res.neff}")

    neff_sel[ip, :] = res.neff
    scene_sel[ip] = scene

    for m in range(M):
        Ex_sel[ip][m] = res.Ex[m]
        Ey_sel[ip][m] = res.Ey[m]

# ---- PLOTS
 
scene0 = scene_sel[0]
polys = scene0["polys"]
grid = scene0["grid"]
x = grid.x
y = grid.y
X, Y = np.meshgrid(x, y)

global_max = 0.0
for ip in range(P):
    for m in range(M):
        global_max = max(global_max, float(np.max(np.abs(Ex_sel[ip][m]))))
        global_max = max(global_max, float(np.max(np.abs(Ey_sel[ip][m]))))
global_max = global_max + 1e-30

vmin = 1e-3
vmax = 1.0
levels = np.linspace(vmin, vmax, 15)

fig, axes = plt.subplots(
    nrows=2*M, ncols=P,
    figsize=(3.3*P, 1.8*(2*M)),
    constrained_layout=True
)

for ip, phi_c_deg in enumerate(phi_list_deg):
    for m in range(M):
        # row indices
        rEx = 2*m
        rEy = 2*m + 1

        Exn = np.abs(Ex_sel[ip][m]) / global_max
        Eyn = np.abs(Ey_sel[ip][m]) / global_max

        ax = axes[rEx, ip]
        cs = ax.contour(X, Y, Exn, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_facecolor("black")
        overlay_polys(ax, polys, colors=colors, lw=1.5, alpha=1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        if m == 0:
            ax.set_title(rf"$\phi_c={phi_c_deg:.0f}^\circ$")
        if ip == 0:
            ax.set_ylabel(f"Mode {m+1}\n|Ex|")

        ax = axes[rEy, ip]
        cs2 = ax.contour(X, Y, Eyn, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_facecolor("black")
        overlay_polys(ax, polys, colors=colors, lw=1.5, alpha=1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        if ip == 0:
            ax.set_ylabel(f"Mode {m+1}\n|Ey|")

mappable = cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
cbar.set_label("Normalized |E|")

fig.suptitle(rf"$\theta_c={theta_c_deg:.0f}^\circ$  (4 modes, |E_x| and |E_y|)", y=1.01)
plt.show()

phi = np.array(phi_list_deg, dtype=float)

plt.figure(figsize=(6.5, 4.5))
markers = ["o", "o", "x", "*"]  
for m in range(M):
    plt.plot(phi, np.real(neff_sel[:, m]),
             marker=markers[m], linestyle="None", label=f"Mode {m+1}")

plt.xlabel(r"$\phi_c$ (degrees)")
plt.ylabel("Calculated effective index")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
