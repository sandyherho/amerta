"""Simulation logger."""
import logging
import numpy as np
from pathlib import Path


class SimulationLogger:
    def __init__(self, scenario_name, log_dir="logs", verbose=True):
        self.scenario_name = scenario_name
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"
        self.verbose  = verbose
        self.warnings = []; self.errors = []
        self.logger   = logging.getLogger(f"amerta_{scenario_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        fh = logging.FileHandler(self.log_file, mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

    def info(self, m):    self.logger.info(m)
    def debug(self, m):   self.logger.debug(m)

    def warning(self, m):
        self.logger.warning(m); self.warnings.append(m)
        if self.verbose: print(f"  WARNING: {m}")

    def error(self, m):
        self.logger.error(m); self.errors.append(m)
        if self.verbose: print(f"  ERROR: {m}")

    def log_parameters(self, params):
        self.info("="*60)
        self.info(f"PARAMETERS - {params.get('scenario_name', '?')}")
        self.info("="*60)
        for k, v in sorted(params.items()):
            self.info(f"  {k}: {v}")
        self.info("="*60)

    def log_timing(self, t):
        self.info("="*60)
        self.info("TIMING BREAKDOWN")
        self.info("="*60)
        for k, v in sorted(t.items()):
            self.info(f"  {k}: {v:.3f} s")
        self.info("="*60)

    def log_error_summary(self, analytical, t_all):
        """
        Log L1/L2 error norms at every timestep and a final summary.

        Parameters
        ----------
        analytical : dict  – output of compute_analytical + fill_error_norms
        t_all      : array – solver time axis (same length as norm arrays)
        """
        self.info("="*60)
        self.info("ANALYTICAL ERROR NORMS")
        self.info("="*60)

        if not analytical.get('available', False):
            self.info("  No analytical solution available for this case.")
            self.info("="*60)
            return

        l1_h = analytical['l1_h']
        l2_h = analytical['l2_h']
        l1_u = analytical['l1_u']
        l2_u = analytical['l2_u']

        self.info(f"  {'t [s]':>10}  {'L1(h) [m]':>14}  {'L2(h) [m]':>14}"
                  f"  {'L1(u) [m/s]':>14}  {'L2(u) [m/s]':>14}")
        self.info(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")

        for i, t in enumerate(t_all):
            self.info(f"  {t:>10.4f}  {l1_h[i]:>14.6e}  {l2_h[i]:>14.6e}"
                      f"  {l1_u[i]:>14.6e}  {l2_u[i]:>14.6e}")

        self.info("="*60)
        self.info("SUMMARY (final timestep)")
        self.info("="*60)
        self.info(f"  L1(h)  = {l1_h[-1]:.6e} m")
        self.info(f"  L2(h)  = {l2_h[-1]:.6e} m")
        self.info(f"  L1(u)  = {l1_u[-1]:.6e} m/s")
        self.info(f"  L2(u)  = {l2_u[-1]:.6e} m/s")
        self.info(f"  max L1(h) over all t = {float(np.max(l1_h)):.6e} m")
        self.info(f"  max L2(h) over all t = {float(np.max(l2_h)):.6e} m")
        self.info("="*60)

    def finalize(self):
        self.info("="*60)
        self.info("SIMULATION SUMMARY")
        self.info("="*60)
        self.info(f"  ERRORS  : {len(self.errors)}")
        self.info(f"  WARNINGS: {len(self.warnings)}")
        self.info(f"  Log file: {self.log_file}")
        self.info("="*60)
