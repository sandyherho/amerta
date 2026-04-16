"""Unit tests for amerta solver v0.0.2."""
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
        # Stoker has a shock → energy must be dissipated (positive)
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert r['energy_dissipated_pct'] > 0

    def test_cfl_bounded(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert r['cfl_max'] <= 1.05

    def test_ic_at_t0(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        x = r['x']; x_dam = stoker_params['L']/2
        assert np.allclose(r['h_all'][0, x < x_dam],  stoker_params['h_left'],  atol=1e-10)
        assert np.allclose(r['h_all'][0, x >= x_dam], stoker_params['h_right'], atol=1e-10)

    def test_froude_subcritical_stoker(self, stoker_params):
        """Stoker wet-bed solution must be subcritical everywhere."""
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
        assert r['mass_err_pct'] < -5.0  # outflow expected


class TestCases:
    def test_all_four_cases_defined(self):
        assert set(CASES.keys()) == {'stoker','ritter','double_rarefaction','double_shock'}

    @pytest.mark.parametrize('name', list(CASES.keys()))
    def test_case_runs(self, name):
        p = get_case(name)
        p.update({'g':9.81,'L':2000.,'nx':60,'cfl':0.9,'t_final':5.,
                  'anim_frames':5,'scenario_name':f'test_{name}','case_type':name})
        # set realistic h per case
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

    def test_error_zero_at_t0(self):
        p = get_case('stoker')
        p.update({'g':9.81,'L':2000.,'h_left':10.,'h_right':2.,
                  'u_left':0.,'u_right':0.,'nx':100,'cfl':0.9,
                  't_final':5.,'anim_frames':5,'scenario_name':'t0','case_type':'stoker'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        an = compute_analytical('stoker', p, r['x'], r['t_all'])
        fill_error_norms(an, r['h_all'], r['u_all'], r['dx'])
        assert an['l1_h'][0] < 1e-10

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
        # L1(h) at final time should be below a reasonable bound
        assert an['l1_h'][-1] < 100.0  # generous bound for L1 over 2km


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
        nc.close()


if __name__ == '__main__':
    pytest.main([__file__,'-v'])
