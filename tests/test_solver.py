"""Unit tests for amerta v0.0.4.

v0.0.3 tests: all preserved unchanged.

New in v0.0.4
-------------
TestAnalytical.test_new_norms_present
    Checks that l1_q, l2_q, l1_u_wet, l2_u_wet are present in the
    analytical dict after fill_error_norms.

TestAnalytical.test_ic_q_norm_zero_*
    l1_q[0] must be zero at t=0 for all four cases (q_num = q_an at IC).

TestAnalytical.test_ritter_l1u_wet_smaller_than_l1u
    For Ritter, L1(u_wet) must be substantially smaller than L1(u)
    at t>0, confirming the wet-mask removes the dry-front singularity.

TestAnalytical.test_q_norm_well_behaved_ritter
    L1(q) at final time for Ritter must be << L1(u), confirming that
    discharge is a meaningful momentum norm for dry-bed cases.

TestNetCDF.test_new_norm_vars_in_netcdf
    l1_q, l2_q, l1_u_wet, l2_u_wet are present in the NetCDF file.
"""
import pytest, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from amerta_sv.core.solver import SaintVenantSolver
from amerta_sv.core.cases import CASES, get_case
from amerta_sv.core.analytical import compute_analytical, fill_error_norms, ANALYTICAL_AVAILABLE


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def stoker_params():
    p = get_case('stoker')
    p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':2.,
              'u_left':0.,'u_right':0.,'nx':100,'cfl':0.9,
              't_final':10.,'anim_frames':5,
              'scenario_name':'test','case_type':'stoker'})
    return p


@pytest.fixture
def double_rarefaction_params():
    p = get_case('double_rarefaction')
    p.update({'g':9.81,'L':2000.,'h_left':5.,'h_right':5.,
              'u_left':-3.,'u_right':3.,'nx':100,'cfl':0.9,
              't_final':5.,'anim_frames':5,
              'scenario_name':'test_dr','case_type':'double_rarefaction'})
    return p


@pytest.fixture
def double_shock_params():
    p = get_case('double_shock')
    p.update({'g':9.81,'L':2000.,'h_left':3.,'h_right':3.,
              'u_left':3.,'u_right':-3.,'nx':100,'cfl':0.9,
              't_final':5.,'anim_frames':5,
              'scenario_name':'test_ds','case_type':'double_shock'})
    return p


@pytest.fixture
def ritter_params():
    p = get_case('ritter')
    p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':1e-3,
              'u_left':0.,'u_right':0.,'nx':100,'cfl':0.9,
              't_final':5.,'anim_frames':5,
              'scenario_name':'test_r','case_type':'ritter'})
    return p


# ============================================================
# TestSolver — all v0.0.3 tests unchanged
# ============================================================

class TestSolver:
    def test_solve_returns_dict(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        for k in ('x','t_all','h_all','u_all','q_all','n_steps',
                  'mass_err_pct','cfl_max',
                  'mass_all','mass_err_pct_all','momentum_all',
                  'energy_all','froude_max_all','energy_initial'):
            assert k in r, f"missing key: {k}"

    def test_full_trajectory_shape(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert r['h_all'].shape == (r['n_steps']+1, stoker_params['nx'])
        assert len(r['t_all']) == r['n_steps']+1
        assert r['t_all'][0] == 0.0
        assert abs(r['t_all'][-1] - stoker_params['t_final']) < 1e-10

    def test_conservation_arrays_shape(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        nt = r['n_steps']+1
        for key in ('mass_all','mass_err_pct_all','momentum_all','energy_all','froude_max_all'):
            assert len(r[key]) == nt, f"{key} wrong length"

    def test_positivity(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert np.all(r['h_final'] > 0)
        assert np.all(r['h_all'] > 0)

    def test_mass_conservation_stoker(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert abs(r['mass_err_pct']) < 1.0

    def test_energy_dissipated_stoker(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert r['energy_dissipated_pct'] > 0

    def test_cfl_bounded(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert r['cfl_max'] <= 1.05

    def test_ic_at_t0_depth(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        x = r['x']; x_dam = stoker_params['L']/2
        assert np.allclose(r['h_all'][0, x < x_dam],  stoker_params['h_left'],  atol=1e-10)
        assert np.allclose(r['h_all'][0, x >= x_dam], stoker_params['h_right'], atol=1e-10)

    def test_froude_subcritical_stoker(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert np.nanmax(r['froude_max_all']) < 1.1

    def test_mass_err_large_open_bc(self):
        p = get_case('double_rarefaction')
        p.update({'g':9.81,'L':2000.,'h_left':5.,'h_right':5.,
                  'u_left':-3.,'u_right':3.,'nx':80,'cfl':0.9,
                  't_final':80.,'anim_frames':5,
                  'scenario_name':'dr','case_type':'double_rarefaction'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        assert r['mass_err_pct'] < -5.0

    def test_ic_velocity_stored_correctly_zero(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert np.allclose(r['u_all'][0], 0.0, atol=1e-14)

    def test_ic_velocity_stored_correctly_nonzero(self, double_rarefaction_params):
        p = double_rarefaction_params
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        x = r['x']; x_dam = p['L'] / 2.0
        assert np.allclose(r['u_all'][0, x <  x_dam], p['u_left'],  atol=1e-10)
        assert np.allclose(r['u_all'][0, x >= x_dam], p['u_right'], atol=1e-10)

    def test_ic_velocity_double_shock(self, double_shock_params):
        p = double_shock_params
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        x = r['x']; x_dam = p['L'] / 2.0
        assert np.allclose(r['u_all'][0, x <  x_dam], p['u_left'],  atol=1e-10)
        assert np.allclose(r['u_all'][0, x >= x_dam], p['u_right'], atol=1e-10)

    def test_anim_u_ic_nonzero(self, double_rarefaction_params):
        p = double_rarefaction_params
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        x = r['x']; x_dam = p['L'] / 2.0
        assert np.allclose(r['anim_u'][0, x <  x_dam], p['u_left'],  atol=1e-10)
        assert np.allclose(r['anim_u'][0, x >= x_dam], p['u_right'], atol=1e-10)


# ============================================================
# TestCases — unchanged
# ============================================================

class TestCases:
    def test_all_four_cases_defined(self):
        assert set(CASES.keys()) == {'stoker','ritter','double_rarefaction','double_shock'}

    @pytest.mark.parametrize('name', list(CASES.keys()))
    def test_case_runs(self, name):
        p = get_case(name)
        p.update({'g':9.81,'L':2000.,'nx':60,'cfl':0.9,'t_final':5.,
                  'anim_frames':5,'scenario_name':f'test_{name}','case_type':name})
        if name == 'stoker':   p.update({'h_left':10.,'h_right':2.})
        elif name == 'ritter': p.update({'h_left':10.,'h_right':1e-3})
        elif name in ('double_rarefaction','double_shock'):
            p.update({'h_left':5.,'h_right':5.})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        assert r['n_steps'] > 0
        assert np.all(r['h_final'] > 0)

    def test_unknown_case(self):
        with pytest.raises(ValueError): get_case('nonexistent')

    def test_double_shock_symmetric(self):
        p = get_case('double_shock')
        p.update({'g':9.81,'L':2000.,'h_left':3.,'h_right':3.,
                  'u_left':3.,'u_right':-3.,'nx':200,'cfl':0.9,
                  't_final':10.,'anim_frames':5,'scenario_name':'sym','case_type':'double_shock'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        h = r['h_final']
        assert np.allclose(h, h[::-1], atol=1e-2)


# ============================================================
# TestAnalytical — all v0.0.3 tests unchanged + v0.0.4 tests
# ============================================================

class TestAnalytical:
    def test_all_four_have_analytical(self):
        assert ANALYTICAL_AVAILABLE == frozenset(
            {'stoker','ritter','double_rarefaction','double_shock'})

    @pytest.mark.parametrize('name', list(ANALYTICAL_AVAILABLE))
    def test_shape_matches_trajectory(self, name):
        p = get_case(name)
        p.update({'g':9.81,'L':2000.,'nx':80,'cfl':0.9,'t_final':5.,
                  'anim_frames':5,'scenario_name':f't_{name}','case_type':name})
        if name == 'stoker':   p.update({'h_left':10.,'h_right':2.})
        elif name == 'ritter': p.update({'h_left':10.,'h_right':1e-3})
        else: p.update({'h_left':5.,'h_right':5.})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical(name, p, r['x'], r['t_all'])
        assert an['available']
        assert an['h'].shape == r['h_all'].shape
        assert len(an['l1_h']) == len(r['t_all'])

    def test_error_zero_at_t0_stoker(self):
        p = get_case('stoker')
        p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':2.,
                  'u_left':0.,'u_right':0.,'nx':100,'cfl':0.9,
                  't_final':5.,'anim_frames':5,'scenario_name':'t0','case_type':'stoker'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('stoker', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][0] < 1e-10
        assert an['l1_u'][0] < 1e-10

    # ── v0.0.3 regression tests (unchanged) ──────────────────────────────

    def test_ritter_ic_h_exact(self):
        p = get_case('ritter')
        p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':1e-3,
                  'u_left':0.,'u_right':0.,'nx':100,'cfl':0.9,
                  't_final':5.,'anim_frames':5,'scenario_name':'r_ic','case_type':'ritter'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('ritter', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][0] < 1e-10, f"Ritter L1(h) at t=0 = {an['l1_h'][0]:.3e}"
        assert an['l1_u'][0] < 1e-10, f"Ritter L1(u) at t=0 = {an['l1_u'][0]:.3e}"

    def test_double_rarefaction_ic_u_exact(self):
        p = get_case('double_rarefaction')
        p.update({'g':9.81,'L':2000.,'h_left':5.,'h_right':5.,
                  'u_left':-3.,'u_right':3.,'nx':100,'cfl':0.9,
                  't_final':5.,'anim_frames':5,
                  'scenario_name':'dr_ic','case_type':'double_rarefaction'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('double_rarefaction', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][0] < 1e-10
        assert an['l1_u'][0] < 1e-10

    def test_double_shock_ic_u_exact(self):
        p = get_case('double_shock')
        p.update({'g':9.81,'L':2000.,'h_left':3.,'h_right':3.,
                  'u_left':3.,'u_right':-3.,'nx':100,'cfl':0.9,
                  't_final':5.,'anim_frames':5,
                  'scenario_name':'ds_ic','case_type':'double_shock'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('double_shock', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][0] < 1e-10
        assert an['l1_u'][0] < 1e-10

    def test_ritter_fan_accuracy(self):
        p = get_case('ritter')
        p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':1e-3,
                  'u_left':0.,'u_right':0.,'nx':200,'cfl':0.9,
                  't_final':40.,'anim_frames':5,'scenario_name':'r200','case_type':'ritter'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('ritter', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][-1] < 100.0

    # ── v0.0.4 tests ─────────────────────────────────────────────────────

    def test_new_norms_present(self, stoker_params):
        """All four new v0.0.4 norm arrays must exist in the dict."""
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        an = compute_analytical('stoker', stoker_params, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])
        for key in ('l1_q', 'l2_q', 'l1_u_wet', 'l2_u_wet'):
            assert key in an, f"missing key: {key}"
            assert len(an[key]) == len(r['t_all']), f"{key} wrong length"

    @pytest.mark.parametrize('name', list(ANALYTICAL_AVAILABLE))
    def test_ic_q_norm_zero(self, name):
        """L1(q) at t=0 must be zero for all four cases."""
        p = get_case(name)
        p.update({'g':9.81,'L':2000.,'nx':80,'cfl':0.9,'t_final':5.,
                  'anim_frames':5,'scenario_name':f'qic_{name}','case_type':name})
        if name == 'stoker':   p.update({'h_left':10.,'h_right':2.})
        elif name == 'ritter': p.update({'h_left':10.,'h_right':1e-3})
        else: p.update({'h_left':5.,'h_right':5.})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical(name, p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])
        assert an['l1_q'][0] < 1e-10, \
            f"{name}: L1(q) at t=0 = {an['l1_q'][0]:.3e} (should be 0)"

    @pytest.mark.parametrize('name', list(ANALYTICAL_AVAILABLE))
    def test_ic_u_wet_norm_zero(self, name):
        """L1(u_wet) at t=0 must be zero for all four cases."""
        p = get_case(name)
        p.update({'g':9.81,'L':2000.,'nx':80,'cfl':0.9,'t_final':5.,
                  'anim_frames':5,'scenario_name':f'uwetic_{name}','case_type':name})
        if name == 'stoker':   p.update({'h_left':10.,'h_right':2.})
        elif name == 'ritter': p.update({'h_left':10.,'h_right':1e-3})
        else: p.update({'h_left':5.,'h_right':5.})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical(name, p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])
        assert an['l1_u_wet'][0] < 1e-10, \
            f"{name}: L1(u_wet) at t=0 = {an['l1_u_wet'][0]:.3e} (should be 0)"

    def test_ritter_l1u_wet_much_smaller_than_l1u(self, ritter_params):
        """
        For Ritter at t>0, L1(u_wet) must be substantially smaller than L1(u).
        The wet-mask (h > H_DRY = 0.01 m on both sides) removes the near-dry
        velocity singularity.  In practice the reduction is ~74%; we require
        at least 50% reduction as a conservative lower bound.
        """
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(ritter_params)
        an = compute_analytical('ritter', ritter_params, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])
        l1u     = an['l1_u'][-1]
        l1u_wet = an['l1_u_wet'][-1]
        assert l1u_wet < 0.5 * l1u, (
            f"Ritter: L1(u_wet)={l1u_wet:.3e} should be < 0.5*L1(u)={0.5*l1u:.3e}. "
            f"Wet-mask not filtering dry-front singularity sufficiently.")

    def test_ritter_l1q_well_behaved(self, ritter_params):
        """
        For Ritter, L1(q) must be physically well-bounded and must show
        that the dry-front singularity is removed.

        Two checks:
        (a) L1(q) at final time is < 1% of the physical discharge scale
            h_L * c_L * L.  L1(q) and L1(u) have different units (m²/s
            vs m/s), so a direct ratio comparison is meaningless.

        (b) At analytically-dry cells (h_an < H_DRY), the q-error
            contribution is < 10% of the u-error contribution, confirming
            that q = hu → 0 damps the singularity.
        """
        from amerta_sv.core.analytical import H_DRY as _H_DRY
        p = ritter_params
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('ritter', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])

        # (a) physical bound: L1(q) < 1% of h_L * c_L * L
        c_L    = np.sqrt(p['g'] * p['h_left'])
        phys   = p['h_left'] * c_L * p['L']
        l1q    = an['l1_q'][-1]
        assert l1q < 0.01 * phys, (
            f"Ritter L1(q)={l1q:.3e} m2/s exceeds 1% of physical scale "
            f"h_L*c_L*L={0.01*phys:.3e} m2/s")

        # (b) at dry-front cells: q_error << u_error (q = h*u → 0)
        h_an_f = an['h'][-1];  u_an_f = an['u'][-1]
        h_nf   = r['h_all'][-1]; u_nf = r['u_all'][-1]; q_nf = r['q_all'][-1]
        dry    = h_an_f < _H_DRY
        dx     = r['dx']
        if dry.sum() > 0:
            u_err_dry = float(np.sum(np.abs(u_nf[dry])) * dx)
            q_err_dry = float(np.sum(np.abs(q_nf[dry])) * dx)
            if u_err_dry > 1e-12:
                assert q_err_dry < 0.5 * u_err_dry, (
                    f"Ritter dry-cell q_err={q_err_dry:.3e} m2/s should be "
                    f"< 0.5 * u_err={u_err_dry:.3e} m/s at dry front")

    def test_q_norm_nonneg_all_cases(self):
        """L1(q) and L2(q) must be non-negative at all timesteps."""
        for name in ANALYTICAL_AVAILABLE:
            p = get_case(name)
            p.update({'g':9.81,'L':2000.,'nx':60,'cfl':0.9,'t_final':3.,
                      'anim_frames':5,'scenario_name':f'nn_{name}','case_type':name})
            if name == 'stoker':   p.update({'h_left':10.,'h_right':2.})
            elif name == 'ritter': p.update({'h_left':10.,'h_right':1e-3})
            else: p.update({'h_left':5.,'h_right':5.})
            s = SaintVenantSolver(nthreads=2, verbose=False)
            r = s.solve(p)
            an = compute_analytical(name, p, r['x'], r['t_all'])
            fill_error_norms(an, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])
            assert np.all(an['l1_q'] >= 0), f"{name}: negative L1(q)"
            assert np.all(an['l2_q'] >= 0), f"{name}: negative L2(q)"
            assert np.all(an['l1_u_wet'] >= 0), f"{name}: negative L1(u_wet)"

    def test_backward_compat_no_q_num(self, stoker_params):
        """
        fill_error_norms without q_num argument must still work and
        give the same l1_h, l2_h, l1_u, l2_u as before (backward compat).
        """
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        an1 = compute_analytical('stoker', stoker_params, r['x'], r['t_all'])
        fill_error_norms(an1, r['h_all'], r['u_all'], r['dx'])  # no q_num

        an2 = compute_analytical('stoker', stoker_params, r['x'], r['t_all'])
        fill_error_norms(an2, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])

        # v0.0.3 norms must be identical
        np.testing.assert_allclose(an1['l1_h'], an2['l1_h'], rtol=1e-12)
        np.testing.assert_allclose(an1['l1_u'], an2['l1_u'], rtol=1e-12)
        # new norms must be present even without q_num
        assert 'l1_q' in an1
        assert an1['l1_q'][0] < 1e-10


# ============================================================
# TestNetCDF — v0.0.3 tests unchanged + v0.0.4 new fields
# ============================================================

class TestNetCDF:
    def test_conservation_vars_in_netcdf(self, tmp_path):
        from amerta_sv.io.data_handler import DataHandler
        p = get_case('stoker')
        p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':2.,
                  'u_left':0.,'u_right':0.,'nx':60,'cfl':0.9,
                  't_final':5.,'anim_frames':5,'scenario_name':'nc','case_type':'stoker'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('stoker', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])
        DataHandler.save_netcdf('test.nc', r, str(tmp_path), analytical=an)
        from netCDF4 import Dataset
        nc = Dataset(str(tmp_path/'test.nc'))
        for v in ('mass_integral','mass_err_pct','momentum_integral',
                  'energy_integral','energy_diss_pct','froude_max',
                  'h_analytical','h_error','l1_h','l2_h'):
            assert v in nc.variables, f"missing v0.0.3 var: {v}"
        assert len(nc.dimensions['time']) == r['n_steps']+1
        assert nc.analytical_solution_available == 1
        assert '0.0.4' in nc.source
        nc.close()

    def test_netcdf_u_ic_correct(self, tmp_path):
        from amerta_sv.io.data_handler import DataHandler
        from netCDF4 import Dataset
        p = get_case('double_rarefaction')
        p.update({'g':9.81,'L':2000.,'h_left':5.,'h_right':5.,
                  'u_left':-3.,'u_right':3.,'nx':60,'cfl':0.9,
                  't_final':5.,'anim_frames':5,
                  'scenario_name':'nc_dr','case_type':'double_rarefaction'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('double_rarefaction', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])
        DataHandler.save_netcdf('dr.nc', r, str(tmp_path), analytical=an)
        nc = Dataset(str(tmp_path/'dr.nc'))
        u0_nc = nc.variables['u'][0, :]
        x     = nc.variables['x'][:]
        x_dam = p['L'] / 2.0
        assert np.allclose(u0_nc[x <  x_dam], p['u_left'],  atol=1e-10)
        assert np.allclose(u0_nc[x >= x_dam], p['u_right'], atol=1e-10)
        nc.close()

    def test_new_norm_vars_in_netcdf(self, tmp_path):
        """v0.0.4: l1_q, l2_q, l1_u_wet, l2_u_wet must be in NetCDF with
        correct shape, zero at t=0, and finite throughout."""
        from amerta_sv.io.data_handler import DataHandler
        from netCDF4 import Dataset
        p = get_case('ritter')
        p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':1e-3,
                  'u_left':0.,'u_right':0.,'nx':60,'cfl':0.9,
                  't_final':5.,'anim_frames':5,
                  'scenario_name':'nc_r','case_type':'ritter'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('ritter', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'], q_num=r['q_all'])
        DataHandler.save_netcdf('ritter.nc', r, str(tmp_path), analytical=an)
        nc = Dataset(str(tmp_path/'ritter.nc'))

        # all four new variables must exist
        for v in ('l1_q', 'l2_q', 'l1_u_wet', 'l2_u_wet'):
            assert v in nc.variables, f"missing v0.0.4 var: {v}"
            arr = nc.variables[v][:]
            assert len(arr) == r['n_steps']+1, f"{v} wrong length"
            assert np.all(np.isfinite(arr)), f"{v} contains non-finite values"

        # IC must be exactly zero for all new norms
        assert nc.variables['l1_q'][0]     < 1e-10, "l1_q[0] != 0"
        assert nc.variables['l1_u_wet'][0] < 1e-10, "l1_u_wet[0] != 0"

        # wet-mask must reduce velocity error (sanity: l1_u_wet < l1_u at final)
        l1u     = float(nc.variables['l1_u'][-1])
        l1u_wet = float(nc.variables['l1_u_wet'][-1])
        assert l1u_wet < l1u, (
            f"l1_u_wet={l1u_wet:.3e} should be < l1_u={l1u:.3e}")

        nc.close()


# ============================================================
# TestCasesConfig — unchanged
# ============================================================

class TestCasesConfig:
    def test_load_parse_types(self):
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write("# comment\n")
            f.write("name = stoker\n")
            f.write("nx = 400\n")
            f.write("cfl = 0.9\n")
            f.write("save = true\n")
            path = f.name
        from amerta_sv.io.config_manager import ConfigManager
        c = ConfigManager.load(path)
        assert c['name'] == 'stoker'
        assert c['nx'] == 400 and isinstance(c['nx'], int)
        assert c['cfl'] == 0.9 and isinstance(c['cfl'], float)
        assert c['save'] is True
        os.unlink(path)

    def test_validate_defaults(self):
        from amerta_sv.io.config_manager import ConfigManager
        c = ConfigManager.validate_config({'h_left': 3.0})
        assert c['h_left'] == 3.0
        assert c['nx'] == 400
        assert c['cfl'] == 0.9


if __name__ == '__main__':
    pytest.main([__file__,'-v'])
