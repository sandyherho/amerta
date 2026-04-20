# `amerta`: A Python Library for 1D Idealized Saint-Venant Dam-Break Simulation

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/amerta.svg)](https://pypi.org/project/amerta/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/1210460312.svg)](https://doi.org/10.5281/zenodo.19596023)

## Model

The Saint-Venant (shallow water) equations in conservative form on a horizontal, frictionless bed:

**Mass conservation:**  ∂ₜh + ∂ₓ(hu) = 0

**Momentum conservation:**  ∂ₜ(hu) + ∂ₓ(hu² + ½gh²) = 0

### Canonical Riemann Problems

| Case | State | Wave Structure |
|:----:|:------|:---------------|
| **1. Stoker** | Wet–wet | Left rarefaction + right shock |
| **2. Ritter** | Wet–dry | Single rarefaction, dry front |
| **3. Double Rarefaction** | Diverging | Two rarefactions |
| **4. Double Shock** | Converging | Two shocks |

## Numerical Method

- **MUSCL reconstruction** (minmod slope limiter, 2nd-order spatial)
- **HLLC approximate Riemann solver** (Roe-averaged wave speeds)
- **SSP-RK2** strong-stability-preserving time integration (Shu & Osher 1988)
- **Adaptive CFL-limited time step** with positivity preservation
- **Numba JIT** acceleration with parallel `prange` sweeps

## Installation

```bash
pip install amerta
```

## Quick Start

**CLI:**
```bash
amerta case1                     # Stoker wet dam break
amerta case2                     # Ritter dry dam break
amerta case3                     # Double rarefaction
amerta case4                     # Double shock
amerta --all                     # Run all four cases
```

**Python API:**
```python
from amerta_sv import SaintVenantSolver, get_case
from amerta_sv.io import ConfigManager, DataHandler
from amerta_sv.core.analytical import compute_analytical, fill_error_norms

cfg = ConfigManager.validate_config({
    **get_case('stoker'),
    'nx': 500, 'cfl': 0.9, 'g': 9.81, 't_final': 80.0,
    'h_left': 10.0, 'h_right': 2.0, 'L': 2000.0,
    'scenario_name': 'stoker', 'case_type': 'stoker'
})

solver = SaintVenantSolver(nthreads=8, verbose=True)
result = solver.solve(cfg)

an = compute_analytical('stoker', cfg, result['x'], result['t_all'])
fill_error_norms(an, result['h_all'], result['u_all'], result['dx'],
                 q_num=result['q_all'])   # pass q_all for best accuracy

print(f"L1(h)     at t_final = {an['l1_h'][-1]:.4e} m")
print(f"L1(q)     at t_final = {an['l1_q'][-1]:.4e} m2/s")
print(f"L1(u_wet) at t_final = {an['l1_u_wet'][-1]:.4e} m/s")
```


## Output Files

For each case, amerta generates:
- `<case>.nc` — CF-1.8 NetCDF4, full trajectory at every timestep including all error norms
- `<case>_metrics.csv` — scalar diagnostics
- `comparison_metrics.csv` — side-by-side comparison across all runs
- `<case>_time_evolution.png`, `<case>_physical.png`, `<case>_numerical.png`
- `<case>.gif` — animated evolution

## Dependencies

numpy ≥ 1.20, scipy ≥ 1.7, matplotlib ≥ 3.3, netCDF4 ≥ 1.5, numba ≥ 0.53, pandas ≥ 1.3, pillow ≥ 8.0, tqdm ≥ 4.60

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
  version = {0.0.3},
  url     = {https://github.com/sandyherho/amerta}
}
```
