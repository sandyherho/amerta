"""Config file parser."""
from pathlib import Path

class ConfigManager:
    @staticmethod
    def load(path):
        p = Path(path)
        if not p.exists(): raise FileNotFoundError(path)
        cfg = {}
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith('#') or '=' not in ln: continue
                k, v = ln.split('=', 1); k = k.strip(); v = v.strip()
                if '#' in v: v = v.split('#')[0].strip()
                cfg[k] = ConfigManager._parse(v)
        return cfg
    @staticmethod
    def _parse(v):
        if v.lower() in ('true','false'): return v.lower()=='true'
        try:
            return float(v) if ('.' in v or 'e' in v.lower()) else int(v)
        except ValueError:
            return v
    @staticmethod
    def validate_config(cfg):
        d = {'scenario_name':'Saint-Venant','case_type':'stoker','g':9.81,'L':100.0,
             'h_left':2.0,'h_right':0.5,'u_left':0.0,'u_right':0.0,
             'nx':400,'cfl':0.9,'t_final':6.0,'n_snaps':5,
             'anim_frames':80,'anim_fps':15,'nthreads':0,'output_dir':'outputs',
             'save_netcdf':True,'save_csv':True,'save_animation':True,'save_diagnostics':True}
        for k,v in d.items():
            if k not in cfg: cfg[k] = v
        return cfg
