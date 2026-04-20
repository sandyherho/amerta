#!/usr/bin/env python
"""CLI for amerta Saint-Venant 1D solver."""
import argparse, sys
from pathlib import Path
from .core.solver import SaintVenantSolver
from .core.cases import CASES, get_case
from .core.analytical import compute_analytical, fill_error_norms
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    print("\n" + "="*70)
    print(" "*14 + "amerta: 1D Saint-Venant Equations Solver")
    print(" "*19 + "MUSCL-HLLC + SSP-RK2")
    print(" "*26 + "v0.0.3")
    print("="*70 + "\n")


def normalize(name):
    s = name.lower().replace(' - ','_').replace(' ','_').replace('-','_')
    while '__' in s: s = s.replace('__','_')
    return s.rstrip('_')


def run_scenario(config, output_dir='outputs', verbose=True, nthreads=None):
    config    = ConfigManager.validate_config(config)
    scenario  = config.get('scenario_name', 'simulation')
    clean     = normalize(scenario)
    case_type = config.get('case_type', '')

    if verbose:
        print(f"\n{'='*60}\nSCENARIO: {scenario}\n{'='*60}")

    logger = SimulationLogger(clean, "logs", verbose)
    timer  = Timer(); timer.start('total')

    try:
        logger.log_parameters(config)
        nt = nthreads if nthreads is not None else config.get('nthreads', 0)

        with timer.time_section('solver_init'):
            if verbose: print("\n[1/4] Initializing solver...")
            solver = SaintVenantSolver(
                nthreads=nt if nt > 0 else None,
                verbose=verbose, logger=logger)

        with timer.time_section('solve'):
            if verbose: print("\n[2/4] Solving Saint-Venant equations...")
            result = solver.solve(config)
            logger.info(
                f"steps={result['n_steps']}  cfl_max={result['cfl_max']:.4f}  "
                f"mass_err={result['mass_err_pct']:+.3e}%")

        with timer.time_section('analytical'):
            if verbose: print("\n[3/4] Computing analytical solution (if available)...")
            analytical = compute_analytical(
                case_type, config, result['x'], result['t_all'])
            fill_error_norms(
                analytical, result['h_all'], result['u_all'], result['dx'])
            logger.log_error_summary(analytical, result['t_all'])
            if analytical['available'] and verbose:
                print(f"  L1(h) at t_final = {analytical['l1_h'][-1]:.4e} m")
                print(f"  L2(h) at t_final = {analytical['l2_h'][-1]:.4e} m")
            elif verbose:
                print(f"  No analytical solution for case_type='{case_type}'")

        if config.get('save_netcdf', True) or config.get('save_csv', True):
            with timer.time_section('save_data'):
                if verbose: print("\n[4/4] Saving outputs...")
                if config.get('save_netcdf', True):
                    f = DataHandler.save_netcdf(
                        f"{clean}.nc", result, output_dir,
                        analytical=analytical)
                    if verbose: print(f"      {f}")
                if config.get('save_csv', True):
                    f = DataHandler.save_csv(
                        f"{clean}_metrics.csv", result, output_dir)
                    if verbose: print(f"      {f}")
                    DataHandler.append_comparison_csv(
                        "comparison_metrics.csv", result, output_dir)

        if config.get('save_animation', True) or config.get('save_diagnostics', True):
            with timer.time_section('visualization'):
                if verbose: print("      Creating visualizations...")
                if config.get('save_diagnostics', True):
                    Animator.fig_time_evolution(
                        result, f"{clean}_time_evolution.png", output_dir, scenario)
                    Animator.fig_physical(
                        result, f"{clean}_physical.png", output_dir, scenario)
                    Animator.fig_numerical(
                        result, f"{clean}_numerical.png", output_dir, scenario)
                if config.get('save_animation', True):
                    Animator.create_gif(
                        result, f"{clean}.gif", output_dir, scenario,
                        fps=int(config.get('anim_fps', 15)))

        timer.stop('total')
        logger.log_timing(timer.get_times())
        if verbose:
            print(f"\n{'='*60}\nCOMPLETED  "
                  f"total={timer.times.get('total',0):.2f}s  "
                  f"solve={timer.times.get('solve',0):.2f}s\n{'='*60}\n")

        result['analytical'] = analytical
        return result

    except Exception as e:
        logger.error(str(e)); raise
    finally:
        logger.finalize()


def main():
    p = argparse.ArgumentParser(
        description="amerta: 1D Saint-Venant solver.",
        epilog="Example: amerta case1 --nthreads 8")
    p.add_argument('case', nargs='?',
                   choices=['case1','case2','case3','case4'],
                   help='case1=stoker, case2=ritter, '
                        'case3=double_rarefaction, case4=double_shock')
    p.add_argument('--config', '-c', type=str)
    p.add_argument('--all',    '-a', action='store_true')
    p.add_argument('--output-dir', '-o', type=str, default='outputs')
    p.add_argument('--nthreads', type=int, default=None)
    p.add_argument('--quiet',  '-q', action='store_true')
    args = p.parse_args()
    verbose = not args.quiet
    if verbose: print_header()

    if args.config:
        run_scenario(ConfigManager.load(args.config),
                     args.output_dir, verbose, args.nthreads)
    elif args.all:
        cdir  = Path(__file__).parent.parent.parent / 'configs'
        files = sorted(cdir.glob('case*.txt'))
        if not files: print("ERROR: no config files found"); sys.exit(1)
        for i, cf in enumerate(files, 1):
            if verbose: print(f"\n[Case {i}/{len(files)}] {cf.stem}")
            run_scenario(ConfigManager.load(str(cf)),
                         args.output_dir, verbose, args.nthreads)
    elif args.case:
        m    = {'case1':'case1_stoker','case2':'case2_ritter',
                'case3':'case3_double_rarefaction','case4':'case4_double_shock'}
        cdir = Path(__file__).parent.parent.parent / 'configs'
        cf   = cdir / f"{m[args.case]}.txt"
        if not cf.exists():
            print(f"ERROR: config not found: {cf}"); sys.exit(1)
        run_scenario(ConfigManager.load(str(cf)),
                     args.output_dir, verbose, args.nthreads)
    else:
        p.print_help(); sys.exit(0)


if __name__ == '__main__':
    main()
