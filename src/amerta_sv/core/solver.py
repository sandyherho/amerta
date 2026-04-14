"""
Saint-Venant 1D solver: MUSCL-HLLC + SSP-RK2.

Numerical method
----------------
    Spatial  : MUSCL reconstruction (minmod limiter) + HLLC Riemann solver
    Temporal : SSP-RK2 (Heun's method, Shu-Osher 1988)
    BCs      : transmissive (zero-gradient)
    Stability: adaptive CFL-limited dt
"""
import os
import numpy as np
from tqdm import tqdm

try:
    import numba
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*a, **k):
        def d(f): return f
        return d if not a else d(a[0])
    prange = range
    def set_num_threads(n): pass


@njit(cache=True)
def _minmod(a, b):
    if a*b <= 0.0: return 0.0
    return a if abs(a) <= abs(b) else b


@njit(cache=True)
def _hllc_flux(hL, qL, hR, qR, g):
    uL = qL/hL; uR = qR/hR
    cL = np.sqrt(g*hL); cR = np.sqrt(g*hR)
    sqL = np.sqrt(hL); sqR = np.sqrt(hR)
    u_roe = (sqL*uL + sqR*uR)/(sqL+sqR)
    c_roe = np.sqrt(0.5*g*(hL+hR))
    SL = min(uL-cL, u_roe-c_roe); SR = max(uR+cR, u_roe+c_roe)
    fhL=qL; fqL=uL*qL+0.5*g*hL*hL
    fhR=qR; fqR=uR*qR+0.5*g*hR*hR
    if SL >= 0.0: return fhL, fqL
    if SR <= 0.0: return fhR, fqR
    num = SL*hR*(uR-SR) - SR*hL*(uL-SL)
    den = hR*(uR-SR) - hL*(uL-SL)
    if abs(den) < 1e-14:
        inv = 1.0/(SR-SL)
        return ((SR*fhL - SL*fhR + SL*SR*(hR-hL))*inv,
                (SR*fqL - SL*fqR + SL*SR*(qR-qL))*inv)
    Ss = num/den
    if Ss >= 0.0:
        hs = hL*(SL-uL)/(SL-Ss); qs = hs*Ss
        return fhL+SL*(hs-hL), fqL+SL*(qs-qL)
    else:
        hs = hR*(SR-uR)/(SR-Ss); qs = hs*Ss
        return fhR+SR*(hs-hR), fqR+SR*(qs-qR)


@njit(parallel=True, cache=True)
def _compute_rhs(h, q, dx, g):
    nx = h.shape[0]; inv_dx = 1.0/dx; nf = nx - 1
    hL=np.empty(nf); hR=np.empty(nf); qL=np.empty(nf); qR=np.empty(nf)
    for i in prange(nf):
        if i == 0:
            sh_L=(h[1]-h[0])*inv_dx; sq_L=(q[1]-q[0])*inv_dx
        else:
            sh_L=_minmod((h[i]-h[i-1])*inv_dx,(h[i+1]-h[i])*inv_dx)
            sq_L=_minmod((q[i]-q[i-1])*inv_dx,(q[i+1]-q[i])*inv_dx)
        if i == nf-1:
            sh_R=(h[nx-1]-h[nx-2])*inv_dx; sq_R=(q[nx-1]-q[nx-2])*inv_dx
        else:
            sh_R=_minmod((h[i+1]-h[i])*inv_dx,(h[i+2]-h[i+1])*inv_dx)
            sq_R=_minmod((q[i+1]-q[i])*inv_dx,(q[i+2]-q[i+1])*inv_dx)
        hL[i]=max(h[i]+0.5*dx*sh_L,1e-8); hR[i]=max(h[i+1]-0.5*dx*sh_R,1e-8)
        qL[i]=q[i]+0.5*dx*sq_L; qR[i]=q[i+1]-0.5*dx*sq_R
    fh=np.empty(nf); fq=np.empty(nf)
    for i in prange(nf):
        a,b = _hllc_flux(hL[i],qL[i],hR[i],qR[i],g); fh[i]=a; fq[i]=b
    rh=np.zeros(nx); rq=np.zeros(nx)
    for i in prange(1, nx-1):
        rh[i] = -(fh[i]-fh[i-1])*inv_dx
        rq[i] = -(fq[i]-fq[i-1])*inv_dx
    return rh, rq


@njit(cache=True)
def _apply_bcs(h, q):
    h[0]=h[1]; h[-1]=h[-2]; q[0]=q[1]; q[-1]=q[-2]


@njit(cache=True)
def _positivity_fix(h, q):
    EPS = 1e-8
    for k in range(h.shape[0]):
        if h[k] < EPS: h[k]=EPS; q[k]=0.0


@njit(cache=True)
def muscl_hllc_ssprk2_step(h, q, dx, dt, g):
    rh1, rq1 = _compute_rhs(h, q, dx, g)
    h1 = h + dt*rh1; q1 = q + dt*rq1
    _apply_bcs(h1, q1); _positivity_fix(h1, q1)
    rh2, rq2 = _compute_rhs(h1, q1, dx, g)
    hn = 0.5*(h + h1 + dt*rh2); qn = 0.5*(q + q1 + dt*rq2)
    _apply_bcs(hn, qn); _positivity_fix(hn, qn)
    return hn, qn


@njit(cache=True)
def compute_dt(h, q, dx, cfl, g):
    smax = 0.0
    for i in range(h.shape[0]):
        u = q[i]/h[i] if h[i] > 1e-8 else 0.0
        s = abs(u) + np.sqrt(g*h[i])
        if s > smax: smax = s
    return cfl*dx/smax if smax > 0 else 1e-4


class SaintVenantSolver:
    """1D Saint-Venant solver with MUSCL-HLLC + SSP-RK2."""

    def __init__(self, nthreads=None, verbose=True, logger=None):
        self.verbose = verbose; self.logger = logger
        if nthreads is None or nthreads == 0:
            nthreads = os.cpu_count()
        self.nthreads = nthreads
        if NUMBA_AVAILABLE:
            set_num_threads(min(nthreads, os.cpu_count()))
        if verbose:
            print(f"  CPU threads: {self.nthreads}")
            print(f"  Numba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")

    def solve(self, params):
        g=params['g']; L=params['L']; nx=int(params['nx']); cfl=params['cfl']
        t_final=params['t_final']; h_left=params['h_left']; h_right=params['h_right']
        u_left=params.get('u_left',0.0); u_right=params.get('u_right',0.0)
        n_snaps=int(params.get('n_snaps',5)); anim_frames=int(params.get('anim_frames',80))

        x = np.linspace(0.5*(L/nx), L-0.5*(L/nx), nx); dx = L/nx
        x_dam = 0.5*L
        h = np.where(x < x_dam, h_left, h_right).astype(np.float64)
        u0 = np.where(x < x_dam, u_left, u_right).astype(np.float64)
        q = (h*u0).astype(np.float64)

        # JIT warmup
        try:
            muscl_hllc_ssprk2_step(h[:32].copy(), q[:32].copy(), dx, 1e-3, g)
        except Exception: pass

        snap_times = np.linspace(0.0, t_final, n_snaps)
        anim_dt = t_final / max(anim_frames-1, 1)
        h_snaps=[h.copy()]; u_snaps=[np.zeros(nx)]; q_snaps=[q.copy()]; captured=1
        anim_h=[h.copy()]; anim_u=[np.zeros(nx)]; anim_q=[q.copy()]; anim_times=[0.0]
        next_anim = anim_dt
        mass0 = float(np.sum(h)*dx); diag_list=[]

        t=0.0; step=0; dt_min=np.inf; dt_max=0.0
        cfls=[]; mass_errs=[]

        pbar = tqdm(total=100, desc="  Integrating", unit="%",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| t={postfix}',
                    disable=not self.verbose)
        last_pct = 0
        while t < t_final:
            dt = float(compute_dt(h, q, dx, cfl, g))
            if t+dt > t_final: dt = t_final-t
            u_pre = np.where(h>1e-8, q/h, 0.0)
            smax = float(np.max(np.abs(u_pre)+np.sqrt(g*h)))
            cfl_act = smax*dt/dx

            h, q = muscl_hllc_ssprk2_step(h, q, dx, dt, g)
            t += dt; step += 1
            dt_min=min(dt_min,dt); dt_max=max(dt_max,dt)
            cfls.append(cfl_act)
            mass = float(np.sum(h)*dx)
            mass_errs.append((mass-mass0)/mass0*100.0)

            if captured < n_snaps and t >= snap_times[captured]-dt:
                u_now = np.where(h>1e-8, q/h, 0.0)
                h_snaps.append(h.copy()); u_snaps.append(u_now.copy()); q_snaps.append(q.copy())
                captured += 1
            if t >= next_anim - 1e-12:
                u_now = np.where(h>1e-8, q/h, 0.0)
                anim_h.append(h.copy()); anim_u.append(u_now.copy())
                anim_q.append(q.copy()); anim_times.append(t)
                next_anim += anim_dt

            pct = int(100*t/t_final)
            if pct > last_pct:
                pbar.update(pct-last_pct); pbar.set_postfix_str(f"{t:.3f}s"); last_pct=pct
        pbar.close()

        if len(h_snaps) < n_snaps:
            u_now = np.where(h>1e-8, q/h, 0.0)
            h_snaps.append(h.copy()); u_snaps.append(u_now.copy()); q_snaps.append(q.copy())
        if len(anim_h) < anim_frames:
            u_now = np.where(h>1e-8, q/h, 0.0)
            anim_h.append(h.copy()); anim_u.append(u_now.copy())
            anim_q.append(q.copy()); anim_times.append(t)

        return {
            'x': x, 'dx': dx, 'params': params,
            'snap_times': np.linspace(0, t_final, len(h_snaps)),
            'h_snaps': np.array(h_snaps), 'u_snaps': np.array(u_snaps), 'q_snaps': np.array(q_snaps),
            'anim_times': np.array(anim_times), 'anim_h': np.array(anim_h),
            'anim_u': np.array(anim_u), 'anim_q': np.array(anim_q),
            'n_steps': step, 'dt_min': dt_min, 'dt_max': dt_max,
            'cfl_max': float(np.max(cfls)), 'cfl_mean': float(np.mean(cfls)),
            'mass_initial': mass0, 'mass_final': float(np.sum(h)*dx),
            'mass_err_pct': mass_errs[-1], 'max_mass_err': float(np.max(np.abs(mass_errs))),
            'h_final': h, 'q_final': q,
            'numba_enabled': NUMBA_AVAILABLE, 'nthreads': self.nthreads,
        }
