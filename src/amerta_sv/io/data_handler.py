"""NetCDF and CSV output 

NetCDF variables
----------------
Coordinates
    x        (x,)       cell-centre positions [m]
    time     (time,)    every solver timestep including t=0 [s]

Numerical solution
    h        (time,x)   water depth [m]
    u        (time,x)   depth-averaged velocity [m/s]
    q        (time,x)   specific discharge [m²/s]

Conservation diagnostics  (time,)
    mass_integral       domain-integrated water volume [m²]
    mass_err_pct        relative mass error vs initial [%]
    momentum_integral   domain-integrated momentum [m³/s]
    energy_integral     domain-integrated total specific energy [m³/s²]
    energy_diss_pct     cumulative energy dissipation vs initial [%]
    froude_max          max Froude number (h > H_DRY = 0.01 m only) [-]

v0.0.3 error norms (when analytical solution available)
    h_analytical (time,x)   exact depth [m]
    u_analytical (time,x)   exact velocity [m/s]
    h_error      (time,x)   h - h_analytical [m]
    u_error      (time,x)   u - u_analytical [m/s]
    l1_h, l2_h   (time,)    depth L1/L2 error norms [m]
    l1_u, l2_u   (time,)    velocity L1/L2 error norms [m/s]

v0.0.4 error norms (new — when analytical solution available)
    l1_q, l2_q       (time,)  discharge L1/L2 error norms [m²/s]
                              q=hu is the conserved variable; well-behaved
                              for dry-bed cases where l1_u diverges.
    l1_u_wet, l2_u_wet (time,) wet-cell velocity L1/L2 norms [m/s]
                              restricted to cells with h>H_DRY on both sides.
"""
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime


class DataHandler:

    @staticmethod
    def save_netcdf(filename, result, output_dir="outputs", analytical=None):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp  = out / filename
        p   = result['params']

        x  = result['x']
        t  = result['t_all']
        h  = result['h_all']
        u  = result['u_all']
        q  = result['q_all']
        g  = float(p['g'])

        with Dataset(fp, 'w', format='NETCDF4') as nc:
            # ── dimensions ────────────────────────────────────────────────
            nc.createDimension('x',    len(x))
            nc.createDimension('time', len(t))

            # ── coordinates ───────────────────────────────────────────────
            vx = nc.createVariable('x','f8',('x',),zlib=True)
            vx[:] = x; vx.units='m'
            vx.long_name='along_channel_distance'; vx.axis='X'

            vt = nc.createVariable('time','f8',('time',),zlib=True)
            vt[:] = t; vt.units='s'; vt.long_name='time'; vt.axis='T'
            vt.comment = 'Every solver timestep; t[0]=0 is the initial condition'

            # ── numerical solution ─────────────────────────────────────────
            vh = nc.createVariable('h','f8',('time','x'),zlib=True,complevel=4)
            vh[:] = h; vh.units='m'; vh.long_name='water_depth'
            vh.standard_name='depth'

            vu = nc.createVariable('u','f8',('time','x'),zlib=True,complevel=4)
            vu[:] = u; vu.units='m s-1'; vu.long_name='depth_averaged_velocity'

            vq = nc.createVariable('q','f8',('time','x'),zlib=True,complevel=4)
            vq[:] = q; vq.units='m2 s-1'; vq.long_name='specific_discharge'

            # ── conservation diagnostics ───────────────────────────────────
            mass_all  = result['mass_all']
            merr_all  = result['mass_err_pct_all']
            mom_all   = result['momentum_all']
            eng_all   = result['energy_all']
            e0        = result['energy_initial']
            fr_all    = result['froude_max_all']
            ediss     = np.where(e0 > 0, (e0 - eng_all)/e0*100.0, 0.0)

            def _v1(name, data, units, long_name, comment=''):
                vv = nc.createVariable(name,'f8',('time',),zlib=True)
                vv[:] = data; vv.units=units; vv.long_name=long_name
                if comment: vv.comment=comment
                return vv

            _v1('mass_integral', mass_all, 'm2',
                'domain_integrated_water_volume',
                'integral of h over x; conserved with closed BCs')
            _v1('mass_err_pct', merr_all, '%',
                'relative_mass_error_vs_initial',
                '(mass(t)-mass0)/mass0*100; nonzero with open BCs is expected')
            _v1('momentum_integral', mom_all, 'm3 s-1',
                'domain_integrated_momentum',
                'integral of q=hu over x; not conserved in general')
            _v1('energy_integral', eng_all, 'm3 s-2',
                'domain_integrated_total_specific_energy',
                'integral of (0.5*u^2*h + 0.5*g*h^2); decreases at shocks')
            _v1('energy_diss_pct', ediss, '%',
                'cumulative_energy_dissipation_vs_initial',
                '(E0-E(t))/E0*100; positive=dissipation at shocks (correct)')
            _v1('froude_max', fr_all, '1',
                'maximum_froude_number_in_wet_domain',
                f'max(|u|/sqrt(gh)) where h > H_DRY={0.01} m; NaN cells excluded')

            # ── analytical + v0.0.3 error fields ──────────────────────────
            has_an = analytical is not None and analytical.get('available', False)
            if has_an:
                vhan = nc.createVariable('h_analytical','f8',('time','x'),
                                         zlib=True,complevel=4)
                vhan[:]=analytical['h']; vhan.units='m'
                vhan.long_name='exact_water_depth'

                vuan = nc.createVariable('u_analytical','f8',('time','x'),
                                         zlib=True,complevel=4)
                vuan[:]=analytical['u']; vuan.units='m s-1'
                vuan.long_name='exact_depth_averaged_velocity'

                vhe = nc.createVariable('h_error','f8',('time','x'),
                                        zlib=True,complevel=4)
                vhe[:]=h-analytical['h']; vhe.units='m'
                vhe.long_name='depth_error_numerical_minus_analytical'

                vue = nc.createVariable('u_error','f8',('time','x'),
                                        zlib=True,complevel=4)
                vue[:]=u-analytical['u']; vue.units='m s-1'
                vue.long_name='velocity_error_numerical_minus_analytical'

                # v0.0.3 norm variables
                for key, lname, units, cmt in [
                    ('l1_h','L1_depth_error_norm','m',
                     'L1=sum|e_h|dx; well-behaved for all cases'),
                    ('l2_h','L2_depth_error_norm','m',
                     'L2=sqrt(sum e_h^2 dx)'),
                    ('l1_u','L1_velocity_error_norm','m s-1',
                     'L1=sum|e_u|dx; CAUTION: inflated near dry front (Ritter) '
                     'due to h-floor; prefer l1_u_wet or l1_q for dry-bed cases'),
                    ('l2_u','L2_velocity_error_norm','m s-1',
                     'L2=sqrt(sum e_u^2 dx); see l1_u caution'),
                ]:
                    vn = nc.createVariable(key,'f8',('time',),zlib=True)
                    vn[:]=analytical[key]; vn.units=units; vn.long_name=lname
                    vn.comment=cmt

                # v0.0.4 norm variables
                for key, lname, units, cmt in [
                    ('l1_q','L1_discharge_error_norm','m2 s-1',
                     'L1=sum|q_num-q_an|dx where q=hu; recommended primary momentum '
                     'metric; well-behaved for dry-bed cases'),
                    ('l2_q','L2_discharge_error_norm','m2 s-1',
                     'L2=sqrt(sum(q_num-q_an)^2 dx)'),
                    ('l1_u_wet','L1_wet_velocity_error_norm','m s-1',
                     'L1(u) restricted to cells where h_num>H_DRY AND h_an>H_DRY '
                     '(H_DRY=0.01m); removes near-dry singularity from Ritter'),
                    ('l2_u_wet','L2_wet_velocity_error_norm','m s-1',
                     'L2(u_wet); see l1_u_wet comment'),
                ]:
                    vn = nc.createVariable(key,'f8',('time',),zlib=True)
                    vn[:]=analytical[key]; vn.units=units; vn.long_name=lname
                    vn.comment=cmt

            # ── global attributes ──────────────────────────────────────────
            nc.Conventions      = 'CF-1.8'
            nc.title            = f"1D Saint-Venant: {p.get('scenario_name','?')}"
            nc.case_type        = p.get('case_type','?')
            nc.institution      = 'amerta'
            nc.source           = 'amerta (MUSCL-HLLC + SSP-RK2)'
            nc.history          = f"Created {datetime.now().isoformat()}"
            nc.references       = 'Toro (2001); Stoker (1957); Ritter (1892)'
            nc.analytical_solution_available = int(has_an)
            nc.h_dry_threshold  = 0.01
            nc.g = float(p['g']);          nc.h_left  = float(p['h_left'])
            nc.h_right = float(p['h_right']); nc.u_left = float(p.get('u_left',0.0))
            nc.u_right = float(p.get('u_right',0.0)); nc.L = float(p['L'])
            nc.nx = int(p['nx']);          nc.cfl_target = float(p['cfl'])
            nc.t_final = float(p['t_final']); nc.n_steps = int(result['n_steps'])
            nc.cfl_max = float(result['cfl_max'])
            nc.mass_err_pct     = float(result['mass_err_pct'])
            nc.max_mass_err_pct = float(result['max_mass_err'])
            nc.energy_dissipated_pct = float(result['energy_dissipated_pct'])
            nc.numba_enabled    = int(result['numba_enabled'])
            nc.nthreads         = int(result['nthreads'])

        return str(fp)

    @staticmethod
    def save_csv(filename, result, output_dir="outputs"):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out/filename; p = result['params']
        row = {
            'scenario': p.get('scenario_name','?'),
            'case_type': p.get('case_type','?'),
            'nx': int(p['nx']), 'cfl_target': p['cfl'], 'cfl_max': result['cfl_max'],
            'h_left': p['h_left'], 'h_right': p['h_right'],
            'u_left': p.get('u_left',0.0), 'u_right': p.get('u_right',0.0),
            'L': p['L'], 't_final': p['t_final'],
            'n_steps': result['n_steps'],
            'dt_min': result['dt_min'], 'dt_max': result['dt_max'],
            'mass_initial': result['mass_initial'], 'mass_final': result['mass_final'],
            'mass_err_pct': result['mass_err_pct'],
            'max_mass_err_pct': result['max_mass_err'],
            'energy_initial': result['energy_initial'],
            'energy_final': result['energy_final'],
            'energy_dissipated_pct': result['energy_dissipated_pct'],
        }
        pd.DataFrame([row]).to_csv(fp, index=False); return str(fp)

    @staticmethod
    def append_comparison_csv(filename, result, output_dir="outputs"):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out/filename; p = result['params']
        row = {
            'scenario': p.get('scenario_name','?'),
            'case_type': p.get('case_type','?'),
            'nx': int(p['nx']), 'cfl_max': result['cfl_max'],
            'n_steps': result['n_steps'],
            'max_mass_err_pct': result['max_mass_err'],
            'energy_dissipated_pct': result['energy_dissipated_pct'],
            'h_max_final': float(np.max(result['h_final'])),
            'u_max_final': float(np.max(np.abs(
                np.where(result['h_final']>1e-8,
                         result['q_final']/result['h_final'],0.0)))),
        }
        df_new = pd.DataFrame([row])
        if fp.exists():
            df = pd.concat([pd.read_csv(fp), df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(fp, index=False); return str(fp)
