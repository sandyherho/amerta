"""NetCDF and CSV output."""
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime

class DataHandler:
    @staticmethod
    def save_netcdf(filename, result, output_dir="outputs"):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out / filename
        p = result['params']
        x = result['x']; t = result['snap_times']
        h = result['h_snaps']; u = result['u_snaps']; q = result['q_snaps']
        with Dataset(fp, 'w', format='NETCDF4') as nc:
            nc.createDimension('x', len(x)); nc.createDimension('time', len(t))
            vx = nc.createVariable('x','f8',('x',), zlib=True); vx[:]=x
            vx.units='m'; vx.long_name='along_channel_distance'; vx.axis='X'
            vt = nc.createVariable('time','f8',('time',), zlib=True); vt[:]=t
            vt.units='s'; vt.long_name='time'; vt.axis='T'
            vh = nc.createVariable('h','f8',('time','x'), zlib=True, complevel=4); vh[:]=h
            vh.units='m'; vh.long_name='water_depth'; vh.standard_name='depth'
            vu = nc.createVariable('u','f8',('time','x'), zlib=True, complevel=4); vu[:]=u
            vu.units='m s-1'; vu.long_name='depth_averaged_velocity'
            vq = nc.createVariable('q','f8',('time','x'), zlib=True, complevel=4); vq[:]=q
            vq.units='m2 s-1'; vq.long_name='discharge_per_unit_width'
            nc.Conventions = 'CF-1.8'
            nc.title = f"1D Saint-Venant: {p.get('scenario_name','?')}"
            nc.case_type = p.get('case_type','?')
            nc.institution = 'amerta'
            nc.source = 'amerta v0.0.1 (MUSCL-HLLC + SSP-RK2)'
            nc.history = f"Created {datetime.now().isoformat()}"
            nc.references = 'Toro (2001); Stoker (1957); Ritter (1892)'
            nc.g = float(p['g']); nc.h_left = float(p['h_left']); nc.h_right = float(p['h_right'])
            nc.u_left = float(p.get('u_left',0.0)); nc.u_right = float(p.get('u_right',0.0))
            nc.L = float(p['L']); nc.nx = int(p['nx']); nc.cfl_target = float(p['cfl'])
            nc.t_final = float(p['t_final'])
            nc.n_steps = int(result['n_steps']); nc.cfl_max = float(result['cfl_max'])
            nc.mass_err_pct = float(result['mass_err_pct'])
            nc.max_mass_err_pct = float(result['max_mass_err'])
            nc.numba_enabled = int(result['numba_enabled']); nc.nthreads = int(result['nthreads'])
        return str(fp)

    @staticmethod
    def save_csv(filename, result, output_dir="outputs"):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out / filename
        p = result['params']
        row = {
            'scenario': p.get('scenario_name','?'), 'case_type': p.get('case_type','?'),
            'nx': int(p['nx']), 'cfl_target': p['cfl'], 'cfl_max': result['cfl_max'],
            'h_left': p['h_left'], 'h_right': p['h_right'],
            'u_left': p.get('u_left',0.0), 'u_right': p.get('u_right',0.0),
            't_final': p['t_final'], 'n_steps': result['n_steps'],
            'dt_min': result['dt_min'], 'dt_max': result['dt_max'],
            'mass_initial': result['mass_initial'], 'mass_final': result['mass_final'],
            'mass_err_pct': result['mass_err_pct'],
            'max_mass_err_pct': result['max_mass_err'],
        }
        pd.DataFrame([row]).to_csv(fp, index=False); return str(fp)

    @staticmethod
    def append_comparison_csv(filename, result, output_dir="outputs"):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        fp = out / filename; p = result['params']
        row = {'scenario': p.get('scenario_name','?'), 'case_type': p.get('case_type','?'),
               'nx': int(p['nx']), 'cfl_max': result['cfl_max'],
               'n_steps': result['n_steps'], 'max_mass_err_pct': result['max_mass_err'],
               'h_max_final': float(np.max(result['h_final'])),
               'u_max_final': float(np.max(np.abs(np.where(result['h_final']>1e-8,
                                          result['q_final']/result['h_final'],0.0))))}
        df_new = pd.DataFrame([row])
        if fp.exists():
            df = pd.concat([pd.read_csv(fp), df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(fp, index=False); return str(fp)
