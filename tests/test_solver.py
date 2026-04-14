"""Unit tests for amerta solver."""
import pytest, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from amerta_sv.core.solver import SaintVenantSolver, compute_dt, muscl_hllc_ssprk2_step
from amerta_sv.core.cases import CASES, get_case


@pytest.fixture
def stoker_params():
    p = get_case('stoker')
    p.update({'g':9.81, 'nx':100, 'cfl':0.9, 't_final':0.5,
              'n_snaps':3, 'anim_frames':10,
              'scenario_name':'test', 'case_type':'stoker'})
    return p

class TestSolver:
    def test_solve_returns_dict(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        for k in ('x','h_snaps','u_snaps','q_snaps','anim_h','n_steps',
                  'mass_err_pct','cfl_max'):
            assert k in r

    def test_positivity(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert np.all(r['h_final'] > 0)

    def test_mass_conservation(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        # transmissive BCs leak mass only through domain edges — small here
        assert abs(r['mass_err_pct']) < 5.0

    def test_cfl_bounded(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert r['cfl_max'] <= 1.05  # small overshoot allowed

    def test_snap_count(self, stoker_params):
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(stoker_params)
        assert r['h_snaps'].shape[0] >= 2


class TestCases:
    def test_all_four_cases_defined(self):
        assert set(CASES.keys()) == {'stoker','ritter','double_rarefaction','double_shock'}

    @pytest.mark.parametrize('name', list(CASES.keys()))
    def test_case_runs(self, name):
        p = get_case(name)
        p.update({'g':9.81,'nx':80,'cfl':0.9,'t_final':0.2,
                  'n_snaps':3,'anim_frames':5,
                  'scenario_name':f'test_{name}','case_type':name})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        assert r['n_steps'] > 0
        assert np.all(r['h_final'] > 0)

    def test_unknown_case(self):
        with pytest.raises(ValueError):
            get_case('nonexistent')

    def test_double_shock_symmetric(self):
        p = get_case('double_shock')
        p.update({'g':9.81,'nx':200,'cfl':0.9,'t_final':0.3,
                  'n_snaps':3,'anim_frames':5,
                  'scenario_name':'sym','case_type':'double_shock'})
        s = SaintVenantSolver(nthreads=2, verbose=False)
        r = s.solve(p)
        # Symmetric IC → reflection-symmetric solution
        h = r['h_final']; n = len(h)
        assert np.allclose(h, h[::-1], atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__,'-v'])
