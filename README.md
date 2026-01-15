# FOTONX

FOTONX: modular photonics simulation toolbox in Python



\# FOTONX — Photonics Simulation Toolbox (WIP)



\*\*FOTONX\*\* is a modular, research-oriented photonics simulation toolbox focused on \*\*Maxwell-based numerical solvers\*\* and reproducible workflows for integrated photonics and microstructured fibers.



This repository currently includes the first stable module:



\- \*\*vmode\*\*: \*\*full-vectorial FDFD eigenmode solver\*\* (2D cross-sections) with \*\*PML\*\*, \*\*GDS import\*\*, \*\*isotropic + anisotropic permittivity tensors\*\*, and optional \*\*(Ez, Hz) reconstruction\*\*.



More solvers (BPM / CMT / EME / FDTD) are planned as separate modules inside the same toolbox.



---



\## Why this exists



Commercial tools are powerful but often inaccessible for students and small research groups.  

FOTONX aims to provide an \*\*academic open codebase\*\* with:



\- Clear physics + numerics structure

\- Reproducible examples

\- A modular API designed to scale to multiple solvers



---



\## Current features (vmode)



\### Geometry / meshing

\- Import 2D cross-sections from \*\*GDSII\*\*

\- Polygon extraction by `layer\_map`

\- Uniform grid generation with padding

\- Optional \*\*cropped simulation window\*\* for large structures (e.g., PCF)



\### Materials

\- \*\*Isotropic materials from a database\*\* (provider: `db\_iso`)

\- \*\*Manual tensor overrides\*\* (provider: `const\_tensor`)  

&nbsp; Supports off-diagonals: `xy, xz, yz`

\- Non-magnetic assumption by default (`μ = 1`) with diagonal overrides supported



\### Boundary conditions

\- Separable \*\*PML\*\* in x/y



\### Solver

\- Full-vectorial FDFD eigenmodes: `Ex, Ey, Hx, Hy`

\- Optional longitudinal reconstruction: `Ez, Hz` (`z\_fields=True`)

\- Eigen-solve with an effective-index guess `n\_guess`



\### Visualization

\- Polygon plotting and overlays

\- Field plots: `|Ex|`, `|Ey|`, `Re(Ex)`, `Re(Ey)` (phase-fixed)

\- Paper-style contour plots for anisotropic cases



---



\## Repository status



\- vmode: working + validated on multiple test cases  



---



\## Installation



\### Option A — editable install (recommended for development)

```bash

pip install -e .



\### Option B — standard install

pip install .



\## Quick start

python examples/Example\_1\_isotropic\_gds.py


All example scrcipts use portable paths to GDS files stored in:
examples/data/



\## Citation

If you use FOTONX in academic work, please cite:



FOTONX (vmode module)

Daniel Rodríguez Guillén, FOTONX: Photonics Simulation Toolbox (FDFD vmode module), GitHub repository, 2026. Check CITATION.cff

## License

See LICENSE for full terms.



\## Author



Daniel Rodríguez Guillén

PhD student — Photonics / Numerical Methods / Scientific Computing



\## Contact



For collaboration or commercial licensing inquiries:

Open a GitHub issue,

Contact via LinkedIn, or by email.

