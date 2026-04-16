# `amerta`: A Python Library for 1D Idealized Saint-Venant Dam-Break Simulation

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/amerta.svg)](https://pypi.org/project/amerta/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/1210460312.svg)](https://doi.org/10.5281/zenodo.19596023)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%23004B87.svg)](https://unidata.github.io/netcdf4-python/)
[![Numba](https://img.shields.io/badge/Numba-%2300A3E0.svg?logo=numba&logoColor=white)](https://numba.pydata.org/)
[![Pillow](https://img.shields.io/badge/Pillow-%23000000.svg)](https://python-pillow.org/)
[![tqdm](https://img.shields.io/badge/tqdm-%23FFC107.svg)](https://tqdm.github.io/)


<p align="center">
  <img src="https://github.com/sandyherho/amerta/blob/main/.assets/case_1_stoker_wet_dam_break.gif" alt="damBreak" width="800">
</p>


## Model

The Saint-Venant (shallow water) equations describe depth-averaged free-surface flow. In conservative form on a horizontal, frictionless bed:

### Governing Equations

**Mass conservation:**

$$\partial_t h + \partial_x (hu) = 0$$

**Momentum conservation:**

$$\partial_t (hu) + \partial_x \left(hu^2 + \tfrac{1}{2} g h^2\right) = 0$$

where $h(x,t)$ is the water depth, $u(x,t)$ is the depth-averaged velocity, $q = hu$ is the specific discharge, and $g$ is gravitational acceleration.

### Vector Form

In conservative vector form $\partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) = 0$ with

$$\mathbf{U} = \begin{pmatrix} h \\ hu \end{pmatrix}, \qquad \mathbf{F}(\mathbf{U}) = \begin{pmatrix} hu \\ hu^2 + \tfrac{1}{2} g h^2 \end{pmatrix}$$

The Jacobian $\partial \mathbf{F} / \partial \mathbf{U}$ has eigenvalues $\lambda_{1,2} = u \mp c$ where $c = \sqrt{gh}$ is the gravity-wave celerity. The Froude number $\mathrm{Fr} = |u|/c$ classifies subcritical ($\mathrm{Fr} < 1$) vs supercritical ($\mathrm{Fr} > 1$) flow.

### Riemann Invariants (for rarefactions)

Across a left-going rarefaction the invariant $u + 2\sqrt{gh}$ is preserved; across a right-going rarefaction $u - 2\sqrt{gh}$ is preserved. Across a shock, the Rankine–Hugoniot conditions hold with shock speed $S$ satisfying

$$S = \frac{h_\star u_\star - h_R u_R}{h_\star - h_R}$$

### Key Parameters

| Parameter | Symbol | Description | Typical Range |
|:---------:|:------:|:------------|:-------------:|
| g | $g$ | Gravitational acceleration | $9.81 \ \text{m/s}^2$ |
| L | $L$ | Channel length | 10–1000 m |
| h_left | $h_L$ | Initial depth left of dam | 0.5–10 m |
| h_right | $h_R$ | Initial depth right of dam | 0–10 m |
| u_left | $u_L$ | Initial velocity left of dam | $-10$ to $+10$ m/s |
| u_right | $u_R$ | Initial velocity right of dam | $-10$ to $+10$ m/s |
| nx | $N_x$ | Number of grid cells | 200–5000 |
| CFL | $\nu$ | Courant number | 0.5–0.9 |

### Canonical Riemann Problems

The dam-break problem with initial state $\mathbf{U}(x, 0) = \mathbf{U}_L$ for $x < x_{\text{dam}}$ and $\mathbf{U}_R$ otherwise produces four classical wave patterns:

| Case | State | Wave Structure | Characteristics |
|:-----:|:------|:---------------|:----------------|
| **1. Stoker** | Wet–wet dam break | Left rarefaction + right shock | Classical flood wave |
| **2. Ritter** | Wet–dry dam break | Single rarefaction, dry front | Catastrophic collapse onto dry bed |
| **3. Double Rarefaction** | Diverging flow | Two rarefactions | Near-vacuum at center |
| **4. Double Shock** | Converging flow | Two shocks | Frontal collision / surge |

## Numerical Method

### MUSCL Reconstruction (Second-Order Spatial)

Cell-interface states are reconstructed from cell averages $\mathbf{U}_i$ using the minmod slope limiter:

$$\mathbf{U}_{i+1/2}^L = \mathbf{U}_i + \tfrac{1}{2}\,\mathrm{minmod}(\mathbf{U}_i - \mathbf{U}_{i-1},\, \mathbf{U}_{i+1} - \mathbf{U}_i)$$

$$\mathbf{U}_{i+1/2}^R = \mathbf{U}_{i+1} - \tfrac{1}{2}\,\mathrm{minmod}(\mathbf{U}_{i+1} - \mathbf{U}_i,\, \mathbf{U}_{i+2} - \mathbf{U}_{i+1})$$

where $\mathrm{minmod}(a,b) = \tfrac{1}{2}[\mathrm{sgn}(a) + \mathrm{sgn}(b)]\,\min(|a|,|b|)$.

### HLLC Approximate Riemann Solver

The interface numerical flux $\mathbf{F}_{i+1/2}$ is computed by HLLC (Harten–Lax–van Leer Contact) using Roe-averaged wave-speed estimates:

$$S_L = \min(u_L - c_L,\ \tilde{u} - \tilde{c}), \qquad S_R = \max(u_R + c_R,\ \tilde{u} + \tilde{c})$$

with contact wave speed

$$S_\star = \frac{S_L h_R (u_R - S_R) - S_R h_L (u_L - S_L)}{h_R (u_R - S_R) - h_L (u_L - S_L)}$$

### SSP-RK2 Time Integration

The strong-stability-preserving two-stage Runge–Kutta scheme (Shu & Osher, 1988):

$$\mathbf{U}^{(1)} = \mathbf{U}^n + \Delta t\, \mathcal{L}(\mathbf{U}^n)$$

$$\mathbf{U}^{n+1} = \tfrac{1}{2} \mathbf{U}^n + \tfrac{1}{2}\mathbf{U}^{(1)} + \tfrac{1}{2} \Delta t\, \mathcal{L}(\mathbf{U}^{(1)})$$

with CFL-limited time step

$$\Delta t = \nu \cdot \frac{\Delta x}{\max_i (|u_i| + \sqrt{g h_i})}$$

### Diagnostics

| Diagnostic | Formula | Interpretation |
|:------:|:--------|:---------------|
| **Celerity** | $c = \sqrt{gh}$ | Gravity-wave speed |
| **Froude number** | $\mathrm{Fr} = \lvert u \rvert / c$ | Flow regime classifier |
| **Specific energy** | $E = \tfrac{1}{2} u^2 + g h$ | Hydraulic head |
| **Mass balance** | $\int h\, dx$ | Conservation check (closed BC) |
| **CFL actual** | $\nu_{\text{act}} = (\lvert u \rvert + c)\, \Delta t / \Delta x$ | Stability monitor |

### Analytical Solutions (v0.0.2)

Exact solutions are available for all four canonical cases and are computed automatically alongside every simulation run. They are written directly into the NetCDF output for post-processing and convergence analysis.

| Case | Method | Reference |
|:----:|:-------|:----------|
| Ritter | Closed-form similarity solution | Ritter (1892) |
| Stoker | Newton iteration on $h_\star$ (Brent) | Stoker (1957) |
| Double Rarefaction | Closed-form symmetric fan solution | Toro (2001) |
| Double Shock | Newton iteration on $h_\star$ (Brent) | Toro (2001) |


## Installation

**From PyPI:**
```bash
pip install amerta
```

**From source:**
```bash
git clone https://github.com/sandyherho/amerta.git
cd amerta
pip install .
```

**Development installation with Poetry:**
```bash
git clone https://github.com/sandyherho/amerta.git
cd amerta
poetry install
```

## Quick Start

**CLI:**
```bash
amerta case1                     # Stoker wet dam break
amerta case2                     # Ritter dry dam break
amerta case3                     # Double rarefaction
amerta case4                     # Double shock
amerta --all                     # Run all four cases
amerta case1 --nthreads 8        # Use 8 threads
```

**Python API:**
```python
from amerta_sv import SaintVenantSolver, get_case
from amerta_sv.io import ConfigManager, DataHandler
from amerta_sv.core.analytical import compute_analytical, fill_error_norms

# Load preset and override grid resolution
cfg = ConfigManager.validate_config({
    **get_case('stoker'),
    'nx': 800, 'cfl': 0.9, 'g': 9.81,
    'scenario_name': 'stoker_hires',
    'case_type': 'stoker'
})

# Solve
solver = SaintVenantSolver(nthreads=8, verbose=True)
result = solver.solve(cfg)

# Compute analytical solution and error norms
an_snap = compute_analytical('stoker', cfg, result['x'], result['snap_times'])
fill_error_norms(an_snap, result['h_snaps'], result['u_snaps'], result['dx'])

print(f"L1(h) at t_final = {an_snap['l1_h'][-1]:.4e} m")
print(f"L2(h) at t_final = {an_snap['l2_h'][-1]:.4e} m")

# Save (analytical fields written automatically into NetCDF)
an_anim = compute_analytical('stoker', cfg, result['x'], result['anim_times'])
fill_error_norms(an_anim, result['anim_h'], result['anim_u'], result['dx'])
DataHandler.save_netcdf('stoker.nc', result, 'outputs',
                        analytical_snap=an_snap, analytical_anim=an_anim)
```

## Features

- **Second-order MUSCL reconstruction** with minmod slope limiter
- **HLLC approximate Riemann solver** with Roe-averaged wave speeds
- **SSP-RK2** strong-stability-preserving time integration
- **Adaptive CFL-limited time step** with positivity preservation
- **Numba JIT** acceleration with parallel `prange` sweeps (user-selectable thread count)
- **Four canonical Riemann test cases** validated against analytical solutions
- **Exact analytical solutions** for all four cases, auto-computed and saved to NetCDF
- **L1/L2 error norms** at every snapshot and animation frame, ready for convergence studies
- **Full time trajectory** (all animation frames) written to NetCDF, not just snapshots
- **CF-1.8 compliant NetCDF4** output with full trajectory data
- **Dark-themed publication figures** (time evolution, physical interpretation, numerical aspects)
- **Animated GIF** with red-dashed dam reference and real-time counter
- **Configurable scenarios** via plain-text config files
- **Progress bars** via `tqdm` for integration and GIF rendering

## Output Files

The library generates, for each case:

- **CSV files**:
  - `<case>_metrics.csv` — single-run diagnostics (steps, CFL, mass error, etc.)
  - `comparison_metrics.csv` — appended across all runs for side-by-side comparison
- **NetCDF**: `<case>.nc` — CF-1.8 with dimensions `(time × x)` and `(anim_time × x)`, containing:
  - `h`, `u`, `q` — numerical solution at snapshot times
  - `h_anim`, `u_anim`, `q_anim` — full time trajectory at every animation frame
  - `h_analytical`, `u_analytical` — exact solution at snapshot times (when available)
  - `h_error`, `u_error` — pointwise numerical minus analytical error fields
  - `l1_h`, `l2_h`, `l1_u`, `l2_u` — integrated error norms at each snapshot
  - `*_anim` variants of all analytical/error fields along the full trajectory
- **PNG**:
  - `<case>_time_evolution.png` — snapshot overlay at saved times
  - `<case>_physical.png` — depth, velocity, Froude, specific energy
  - `<case>_numerical.png` — celerity, $\Delta t$ range, CFL stats, mass error
- **GIF**: `<case>.gif` — animated evolution with dam reference line and parameter subtitle
- **Log**: `logs/<case>.log` — full run log

## Dependencies

- **numpy** >= 1.20.0
- **scipy** >= 1.7.0
- **matplotlib** >= 3.3.0
- **pandas** >= 1.3.0
- **netCDF4** >= 1.5.0
- **numba** >= 0.53.0
- **Pillow** >= 8.0.0
- **tqdm** >= 4.60.0

## Changelog

### v0.0.2
- Added `analytical.py`: exact solutions for all four canonical Riemann cases (Ritter closed-form, Stoker Newton, double rarefaction closed-form, double shock Newton)
- NetCDF output now stores the full animation-frame trajectory (`h_anim`, `u_anim`, `q_anim`) in addition to snapshots
- Analytical fields (`h_analytical`, `u_analytical`), error fields (`h_error`, `u_error`), and integrated L1/L2 norms written to NetCDF at both snapshot and animation-frame time axes
- Extended test suite with `TestAnalytical` class covering shape, norm finiteness, and IC correctness

### v0.0.1
- Initial release

## License

MIT 2026 © Dasapta E. Irawan, Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Astyka Pamumpuni, Rendy D. Kartiko, Edi Riawan, Rusmawan Suwarman, and Deny J. Puradimaja

## Citation

```bibtex
@software{irawanEtAl2026_amerta,
  title   = {{\texttt{amerta}: A Python library for 1D idealized Saint-Venant dam-break simulation}},
  author  = {Irawan, Dasapta E. and Herho, Sandy H. S. and Anwar, Iwan P. and
             Khadami, Faruq and Pamumpuni, Astyka and Kartiko, Rendy D. and
             Riawan, Edi and Suwarman, Rusmawan and Puradimaja, Deny J.},
  year    = {2026},
  version = {0.0.2},
  url     = {https://github.com/sandyherho/amerta}
}
```
