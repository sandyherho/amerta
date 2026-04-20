"""
Exact analytical solutions for the four canonical 1D SWE Riemann problems.

Supported cases
---------------
ritter            : Ritter (1892) wet-dry, single rarefaction – closed form
stoker            : Stoker (1957) wet-wet, rarefaction + shock – Newton on h*
double_rarefaction: symmetric diverging flow – closed form
double_shock      : symmetric converging flow – Newton on h*

All solvers work in similarity coordinates  ξ = (x − x_dam) / t
and vectorise over (x, t) arrays.

Changelog
---------
v0.0.3 — Bug fix: _ritter_at_t now accepts h_right and sets h[x >= x_dam] = hR
         at t=0, so the analytical initial condition matches the numerical IC
         exactly (h_right = 1e-3 in the config, not 0).  For t > 0 the
         analytical solution still applies the exact Ritter dry-front formula
         (h → 0 ahead of the wave front), which is mathematically correct.
         The dispatch lambda is updated accordingly.
         Previously, L1(h)|_{t=0} was nonzero (~1.0 m) for the Ritter case
         because the analytical IC had h_R = 0 while the numerical IC had
         h_R = 1e-3.
"""
import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Ritter (dry-bed) – fully closed form
# ---------------------------------------------------------------------------

def _ritter_at_t(x, t, hL, hR, g, x_dam):
    """
    Exact Ritter (1892) solution.

    Parameters
    ----------
    hR : float
        Right-side depth used *only* at t = 0 to match the numerical IC.
        For t > 0 the undisturbed right state ahead of the dry front is
        analytically 0; hR is ignored there.
    """
    cL = np.sqrt(g * hL)
    h = np.zeros_like(x)
    u = np.zeros_like(x)
    if t <= 0.0:
        h[x <  x_dam] = hL
        h[x >= x_dam] = hR   # FIX: match numerical IC exactly (was implicitly 0)
        return h, u
    xi = (x - x_dam) / t
    m1 = xi < -cL                       # undisturbed left
    m2 = (~m1) & (xi <= 2.0 * cL)      # rarefaction fan
    h[m1] = hL
    h[m2] = (2.0 * cL - xi[m2]) ** 2 / (9.0 * g)
    u[m2] = (2.0 / 3.0) * (xi[m2] + cL)
    # dry front: h=0, u=0 (already zero-initialised) — analytically exact
    return h, u


# ---------------------------------------------------------------------------
# Stoker (wet-bed) – Newton on h*, then explicit fan/shock construction
# ---------------------------------------------------------------------------

def _stoker_star_state(hL, hR, uL, uR, g):
    """Return (h*, u*, S_shock) for the Stoker problem."""
    cL = np.sqrt(g * hL)

    def residual(hstar):
        cstar = np.sqrt(g * hstar)
        u_left  = uL + 2.0 * (cL - cstar)
        u_right = uR + (hstar - hR) * np.sqrt(g * (hstar + hR) / (2.0 * hstar * hR))
        return u_left - u_right

    hstar = brentq(residual, hR * 1.0001, hL * 0.9999, xtol=1e-12, maxiter=200)
    cstar = np.sqrt(g * hstar)
    ustar = uL + 2.0 * (cL - cstar)
    S     = (hstar * ustar - hR * uR) / (hstar - hR)
    return hstar, ustar, S


def _stoker_at_t(x, t, hL, hR, uL, uR, g, x_dam):
    cL = np.sqrt(g * hL)
    K  = uL + 2.0 * cL
    h = np.zeros_like(x)
    u = np.zeros_like(x)
    if t <= 0.0:
        h[x < x_dam] = hL;  h[x >= x_dam] = hR
        u[x < x_dam] = uL;  u[x >= x_dam] = uR
        return h, u

    hstar, ustar, S = _stoker_star_state(hL, hR, uL, uR, g)
    cstar = np.sqrt(g * hstar)
    xi = (x - x_dam) / t

    m1 = xi < uL - cL
    m2 = (~m1) & (xi <= ustar - cstar)
    m3 = (~m1) & (~m2) & (xi < S)
    m4 = xi >= S

    h[m1] = hL;  u[m1] = uL
    c_fan = (K - xi[m2]) / 3.0
    h[m2] = np.maximum(c_fan ** 2 / g, 0.0)
    u[m2] = xi[m2] + c_fan
    h[m3] = hstar;  u[m3] = ustar
    h[m4] = hR;  u[m4] = uR
    return h, u


# ---------------------------------------------------------------------------
# Symmetric double rarefaction – closed form
# ---------------------------------------------------------------------------

def _double_rarefaction_at_t(x, t, hL, hR, uL, uR, g, x_dam):
    """uL = -U, uR = +U, hL = hR = h0."""
    h0 = hL;  c0 = np.sqrt(g * h0)
    K_L = uL + 2.0 * c0
    K_R = uR - 2.0 * c0
    cstar = K_L / 2.0
    hstar = max(cstar ** 2 / g, 0.0)

    h = np.zeros_like(x)
    u = np.zeros_like(x)
    if t <= 0.0:
        h[x < x_dam] = h0;  h[x >= x_dam] = h0
        u[x < x_dam] = uL;  u[x >= x_dam] = uR
        return h, u

    xi = (x - x_dam) / t

    m1 = xi < uL - c0
    m2 = (~m1) & (xi <= -cstar)
    m3 = (~m1) & (~m2) & (xi < cstar)
    m4 = (~m1) & (~m2) & (~m3) & (xi <= uR + c0)
    m5 = xi > uR + c0

    h[m1] = h0;  u[m1] = uL
    c_L = (K_L - xi[m2]) / 3.0
    h[m2] = np.maximum(c_L ** 2 / g, 0.0)
    u[m2] = xi[m2] + c_L
    h[m3] = hstar;  u[m3] = 0.0
    c_R = (xi[m4] - K_R) / 3.0
    h[m4] = np.maximum(c_R ** 2 / g, 0.0)
    u[m4] = xi[m4] - c_R
    h[m5] = h0;  u[m5] = uR
    return h, u


# ---------------------------------------------------------------------------
# Symmetric double shock – Newton on h*, then explicit shock placement
# ---------------------------------------------------------------------------

def _double_shock_star_h(h0, U, g):
    """Find h* > h0 for symmetric converging-flow double-shock problem."""
    def residual(hstar):
        return U - (hstar - h0) * np.sqrt(g * (hstar + h0) / (2.0 * hstar * h0))
    return brentq(residual, h0 * 1.0001, h0 * 500.0, xtol=1e-12, maxiter=200)


def _double_shock_at_t(x, t, hL, hR, uL, uR, g, x_dam):
    """uL = +U, uR = -U, hL = hR = h0."""
    h0 = hL;  U = uL
    hstar = _double_shock_star_h(h0, U, g)
    S_L = h0 * U / (h0 - hstar)
    S_R = -S_L

    h = np.zeros_like(x)
    u = np.zeros_like(x)
    if t <= 0.0:
        h[x < x_dam] = h0;  h[x >= x_dam] = h0
        u[x < x_dam] = uL;  u[x >= x_dam] = uR
        return h, u

    xi = (x - x_dam) / t

    m1 = xi < S_L
    m2 = (~m1) & (xi <= S_R)
    m3 = xi > S_R

    h[m1] = h0;  u[m1] = U
    h[m2] = hstar
    h[m3] = h0;  u[m3] = uR
    return h, u


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ANALYTICAL_AVAILABLE = frozenset({'ritter', 'stoker', 'double_rarefaction', 'double_shock'})

# v0.0.3: ritter dispatch now passes p['h_right'] so the IC at t=0 is exact.
_DISPATCH = {
    'ritter':             lambda x, t, p, xd: _ritter_at_t(
                              x, t, p['h_left'], p['h_right'], p['g'], xd),
    'stoker':             lambda x, t, p, xd: _stoker_at_t(
                              x, t, p['h_left'], p['h_right'],
                              p.get('u_left', 0.0), p.get('u_right', 0.0), p['g'], xd),
    'double_rarefaction': lambda x, t, p, xd: _double_rarefaction_at_t(
                              x, t, p['h_left'], p['h_right'],
                              p.get('u_left', 0.0), p.get('u_right', 0.0), p['g'], xd),
    'double_shock':       lambda x, t, p, xd: _double_shock_at_t(
                              x, t, p['h_left'], p['h_right'],
                              p.get('u_left', 0.0), p.get('u_right', 0.0), p['g'], xd),
}


def compute_analytical(case_type, params, x, t_array):
    """
    Compute the exact analytical solution for a given case type.

    Parameters
    ----------
    case_type : str
        One of 'ritter', 'stoker', 'double_rarefaction', 'double_shock'.
    params : dict
        Simulation config (needs g, h_left, h_right, u_left, u_right, L).
    x : ndarray, shape (nx,)
        Cell-centre positions [m].
    t_array : ndarray, shape (nt,)
        Times at which to evaluate [s].

    Returns
    -------
    dict with keys:
        'available' : bool
        'h'         : ndarray (nt, nx) or None
        'u'         : ndarray (nt, nx) or None
        'l1_h'      : ndarray (nt,)   — filled by fill_error_norms
        'l2_h'      : ndarray (nt,)
        'l1_u'      : ndarray (nt,)
        'l2_u'      : ndarray (nt,)
    """
    if case_type not in ANALYTICAL_AVAILABLE:
        return {'available': False, 'h': None, 'u': None,
                'l1_h': None, 'l2_h': None, 'l1_u': None, 'l2_u': None}

    x_dam = 0.5 * float(params['L'])
    fn    = _DISPATCH[case_type]
    nt    = len(t_array)
    nx    = len(x)
    h_an  = np.zeros((nt, nx))
    u_an  = np.zeros((nt, nx))

    for i, t in enumerate(t_array):
        h_an[i], u_an[i] = fn(x, float(t), params, x_dam)

    return {
        'available': True,
        'h': h_an,
        'u': u_an,
        'l1_h': np.zeros(nt),
        'l2_h': np.zeros(nt),
        'l1_u': np.zeros(nt),
        'l2_u': np.zeros(nt),
    }


def fill_error_norms(analytical, h_num, u_num, dx):
    """
    Given analytical dict and numerical arrays, fill L1/L2 error norms in-place.

    L1(h) = sum_i |h_num_i - h_an_i| * dx   [m]
    L2(h) = sqrt(sum_i (h_num_i - h_an_i)^2 * dx)   [m]

    Parameters
    ----------
    analytical : dict returned by compute_analytical
    h_num : ndarray (nt, nx)
    u_num : ndarray (nt, nx)
    dx    : float
    """
    if not analytical['available']:
        return
    for i in range(h_num.shape[0]):
        dh = np.abs(h_num[i] - analytical['h'][i])
        du = np.abs(u_num[i] - analytical['u'][i])
        analytical['l1_h'][i] = np.sum(dh) * dx
        analytical['l2_h'][i] = np.sqrt(np.sum(dh ** 2) * dx)
        analytical['l1_u'][i] = np.sum(du) * dx
        analytical['l2_u'][i] = np.sqrt(np.sum(du ** 2) * dx)
