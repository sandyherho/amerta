"""Simulation logger."""
import logging
from pathlib import Path

class SimulationLogger:
    def __init__(self, scenario_name, log_dir="logs", verbose=True):
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"
        self.verbose = verbose
        self.warnings = []; self.errors = []
        self.logger = logging.getLogger(f"amerta_{scenario_name}")
        self.logger.setLevel(logging.DEBUG); self.logger.handlers = []
        fh = logging.FileHandler(self.log_file, mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
    def info(self, m): self.logger.info(m)
    def debug(self, m): self.logger.debug(m)
    def warning(self, m):
        self.logger.warning(m); self.warnings.append(m)
        if self.verbose: print(f"  WARNING: {m}")
    def error(self, m):
        self.logger.error(m); self.errors.append(m)
        if self.verbose: print(f"  ERROR: {m}")
    def log_parameters(self, params):
        self.info("="*60); self.info(f"PARAMETERS - {params.get('scenario_name','?')}"); self.info("="*60)
        for k,v in sorted(params.items()): self.info(f"  {k}: {v}")
        self.info("="*60)
    def log_timing(self, t):
        self.info("="*60); self.info("TIMING BREAKDOWN"); self.info("="*60)
        for k,v in sorted(t.items()): self.info(f"  {k}: {v:.3f} s")
        self.info("="*60)
    def finalize(self):
        self.info("="*60); self.info("SIMULATION SUMMARY"); self.info("="*60)
        self.info(f"  ERRORS: {len(self.errors)}"); self.info(f"  WARNINGS: {len(self.warnings)}")
        self.info(f"  Log file: {self.log_file}"); self.info("="*60)
