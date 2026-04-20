"""Unit tests for amerta solver v0.0.3.

New in v0.0.3
-------------
TestSolver.test_ic_velocity_stored_correctly
    Regression test for the u_all[0] = zeros bug.
    Asserts that u_all[0] exactly equals the specified initial velocity
    field for a case with nonzero ICs (double rarefaction).

TestAnalytical.test_ritter_ic_h_exact
    Regression test for the Ritter t=0 IC mismatch bug.
    Asserts L1(h)|_{t=0} < 1e-10, i.e. the analytical h at t=0
    matches the numerical IC (h_right = 1e-3) exactly.

TestAnalytical.test_double_rarefaction_ic_u_exact
    Regression test that L1(u)|_{t=0} = 0 for double rarefaction.

TestAnalytical.test_double_shock_ic_u_exact
    Regression test that L1(u)|_{t=0} = 0 for double shock.
"""
import pytest, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from amerta_sv.core.solver import SaintVenantSolver
from amerta_sv.core.cases import CASES, get_case
from amerta_sv.core.analytical import compute_analytical, fill_error_norms, ANALYTICAL_AVAILABLE


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
        """Double rarefaction with open BCs should have large mass loss."""
        p = get_case('double_rarefaction')
        p.update({'g':9.81,'L':2000.,'h_left':5.,'h_right':5.,
                  'u_left':-3.,'u_right':3.,'nx':80,'cfl':0.9,
                  't_final':80.,'anim_frames':5,
                  'scenario_name':'dr','case_type':'double_rarefaction'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        assert r['mass_err_pct'] < -5.0

    # ------------------------------------------------------------------
    # v0.0.3 regression: u_all[0] must store actual initial velocity
    # ------------------------------------------------------------------

    def test_ic_velocity_stored_correctly_zero(self, stoker_params):
        """
        For cases where u_left = u_right = 0, u_all[0] must be all zeros.
        This was already correct in v0.0.2 but is kept as a sanity check.
        """
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert np.allclose(r['u_all'][0], 0.0, atol=1e-14)

    def test_ic_velocity_stored_correctly_nonzero(self, double_rarefaction_params):
        """
        v0.0.3 bug fix: u_all[0] must equal the initial velocity field,
        NOT np.zeros(nx).  Previously this test would have failed because
        u_all[0] was silently stored as zeros regardless of u_left/u_right.
        """
        p = double_rarefaction_params
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        x = r['x']; x_dam = p['L'] / 2.0
        assert np.allclose(r['u_all'][0, x <  x_dam], p['u_left'],  atol=1e-10), \
            "u_all[0] left of dam does not match u_left"
        assert np.allclose(r['u_all'][0, x >= x_dam], p['u_right'], atol=1e-10), \
            "u_all[0] right of dam does not match u_right"

    def test_ic_velocity_double_shock(self, double_shock_params):
        """Same regression test for the double shock case."""
        p = double_shock_params
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        x = r['x']; x_dam = p['L'] / 2.0
        assert np.allclose(r['u_all'][0, x <  x_dam], p['u_left'],  atol=1e-10)
        assert np.allclose(r['u_all'][0, x >= x_dam], p['u_right'], atol=1e-10)

    def test_anim_u_ic_nonzero(self, double_rarefaction_params):
        """
        anim_u[0] was also stored as zeros(nx) in v0.0.2.
        In v0.0.3 it must equal u0.
        """
        p = double_rarefaction_params
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        x = r['x']; x_dam = p['L'] / 2.0
        assert np.allclose(r['anim_u'][0, x <  x_dam], p['u_left'],  atol=1e-10)
        assert np.allclose(r['anim_u'][0, x >= x_dam], p['u_right'], atol=1e-10)


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
        """Stoker: both L1(h) and L1(u) must be exactly 0 at t=0."""
        p = get_case('stoker')
        p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':2.,
                  'u_left':0.,'u_right':0.,'nx':100,'cfl':0.9,
                  't_final':5.,'anim_frames':5,'scenario_name':'t0','case_type':'stoker'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('stoker', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][0] < 1e-10, f"L1(h) at t=0 not zero: {an['l1_h'][0]}"
        assert an['l1_u'][0] < 1e-10, f"L1(u) at t=0 not zero: {an['l1_u'][0]}"

    # ------------------------------------------------------------------
    # v0.0.3 regression: L1(h)|_{t=0} and L1(u)|_{t=0} must be zero
    # for ALL four cases after both bug fixes.
    # ------------------------------------------------------------------

    def test_ritter_ic_h_exact(self):
        """
        v0.0.3 fix: _ritter_at_t now returns h_right on the dry side at t=0.
        Previously L1(h)|_{t=0} ≈ 1.0 m because the analytical IC had h_R=0
        while the numerical IC had h_R=1e-3.
        """
        p = get_case('ritter')
        p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':1e-3,
                  'u_left':0.,'u_right':0.,'nx':100,'cfl':0.9,
                  't_final':5.,'anim_frames':5,'scenario_name':'r_ic','case_type':'ritter'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('ritter', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][0] < 1e-10, \
            f"Ritter L1(h) at t=0 not zero (got {an['l1_h'][0]:.3e}). " \
            f"Check _ritter_at_t h_right fix."
        assert an['l1_u'][0] < 1e-10, \
            f"Ritter L1(u) at t=0 not zero (got {an['l1_u'][0]:.3e})."

    def test_double_rarefaction_ic_u_exact(self):
        """
        v0.0.3 fix: u_all[0] now stores actual initial velocities.
        Previously L1(u)|_{t=0} ≈ 6000 m²/s for double rarefaction
        because u_all[0] was all zeros while u_left=-3, u_right=+3.
        """
        p = get_case('double_rarefaction')
        p.update({'g':9.81,'L':2000.,'h_left':5.,'h_right':5.,
                  'u_left':-3.,'u_right':3.,'nx':100,'cfl':0.9,
                  't_final':5.,'anim_frames':5,
                  'scenario_name':'dr_ic','case_type':'double_rarefaction'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('double_rarefaction', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][0] < 1e-10, \
            f"DR L1(h) at t=0 not zero: {an['l1_h'][0]:.3e}"
        assert an['l1_u'][0] < 1e-10, \
            f"DR L1(u) at t=0 not zero: {an['l1_u'][0]:.3e}. " \
            f"u_all[0] bug not fixed."

    def test_double_shock_ic_u_exact(self):
        """
        Same regression test for double shock (u_left=+3, u_right=-3).
        Previously L1(u)|_{t=0} ≈ 6000 m²/s.
        """
        p = get_case('double_shock')
        p.update({'g':9.81,'L':2000.,'h_left':3.,'h_right':3.,
                  'u_left':3.,'u_right':-3.,'nx':100,'cfl':0.9,
                  't_final':5.,'anim_frames':5,
                  'scenario_name':'ds_ic','case_type':'double_shock'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('double_shock', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][0] < 1e-10, \
            f"DS L1(h) at t=0 not zero: {an['l1_h'][0]:.3e}"
        assert an['l1_u'][0] < 1e-10, \
            f"DS L1(u) at t=0 not zero: {an['l1_u'][0]:.3e}. " \
            f"u_all[0] bug not fixed."

    def test_ritter_fan_accuracy(self):
        """Fan interior errors should be small (< 0.2m) for nx=200."""
        p = get_case('ritter')
        p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':1e-3,
                  'u_left':0.,'u_right':0.,'nx':200,'cfl':0.9,
                  't_final':40.,'anim_frames':5,'scenario_name':'r200','case_type':'ritter'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('ritter', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][-1] < 100.0


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
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        DataHandler.save_netcdf('test.nc', r, str(tmp_path), analytical=an)
        from netCDF4 import Dataset
        nc = Dataset(str(tmp_path/'test.nc'))
        for v in ('mass_integral','mass_err_pct','momentum_integral',
                  'energy_integral','energy_diss_pct','froude_max',
                  'h_analytical','h_error','l1_h','l2_h'):
            assert v in nc.variables, f"missing: {v}"
        assert len(nc.dimensions['time']) == r['n_steps']+1
        assert nc.analytical_solution_available == 1
        # check version string updated
        assert '0.0.3' in nc.source
        nc.close()

    def test_netcdf_u_ic_correct(self, tmp_path):
        """
        v0.0.3: verify that u[0, :] stored in NetCDF equals the actual
        initial velocity, not zeros.
        """
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
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        DataHandler.save_netcdf('dr.nc', r, str(tmp_path), analytical=an)
        nc = Dataset(str(tmp_path/'dr.nc'))
        u0_nc = nc.variables['u'][0, :]
        x     = nc.variables['x'][:]
        x_dam = p['L'] / 2.0
        assert np.allclose(u0_nc[x <  x_dam], p['u_left'],  atol=1e-10), \
            "NetCDF u[0,:] left of dam wrong"
        assert np.allclose(u0_nc[x >= x_dam], p['u_right'], atol=1e-10), \
            "NetCDF u[0,:] right of dam wrong"
        nc.close()


if __name__ == '__main__':
    pytest.main([__file__,'-v'])
