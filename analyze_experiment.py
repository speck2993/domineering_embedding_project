"""Asymptotic gap analysis for probe R² between model conditions.

This script analyzes whether the probe R² advantage of embedded models
persists to convergence by fitting asymptotic curves and bootstrapping
confidence intervals.

Usage:
    python analyze_experiment.py [--bootstrap N] [--layer LAYER]

The script reads from results/histories.json and saves plots to plots/.

By default, analyzes layer 2 (the embedded layer where small model weights are copied).
Use --layer -1 for post-final-LN layer if desired.
"""

import json
import argparse
import time
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import torch

# Suppress curve_fit warnings during bootstrap
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Lazy imports for heavy modules (only loaded when needed)
_probing = None
_model = None
_data_loader = None
_training = None


def _import_probing():
    """Lazy import probing module."""
    global _probing
    if _probing is None:
        import probing as _probing
    return _probing


def _import_model():
    """Lazy import model module."""
    global _model
    if _model is None:
        import model as _model
    return _model


def _import_data_loader():
    """Lazy import data_loader module."""
    global _data_loader
    if _data_loader is None:
        import data_loader as _data_loader
    return _data_loader


def _import_training():
    """Lazy import training module."""
    global _training
    if _training is None:
        import training as _training
    return _training


# =============================================================================
# Data Loading
# =============================================================================

def load_histories(path='results/histories.json'):
    """Load training histories from JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_gap_data(histories, cond_a, cond_b, layer='2'):
    """Extract gap trajectories for a pair of conditions.

    Args:
        histories: Dict of training histories
        cond_a: Pattern for condition A (e.g., 'Large+embed(aux)')
        cond_b: Pattern for condition B (e.g., 'Large-baseline')
        layer: Which layer's probe R² to use (default '2' = embedded layer)

    Returns:
        steps: array of probe evaluation steps
        gaps: array of shape (n_runs, n_steps) with gap = R²_a - R²_b
        seeds: list of seed values used
    """
    # Find matching runs for each condition
    runs_a = {k: v for k, v in histories.items() if cond_a in k}
    runs_b = {k: v for k, v in histories.items() if cond_b in k}

    if not runs_a:
        raise ValueError(f"No runs found matching condition '{cond_a}'")
    if not runs_b:
        raise ValueError(f"No runs found matching condition '{cond_b}'")

    # Extract seed numbers and match
    def get_seed(name):
        # Pattern: 'Large+embed(aux)_s42' -> 42
        return int(name.split('_s')[-1])

    seeds_a = {get_seed(k): k for k in runs_a}
    seeds_b = {get_seed(k): k for k in runs_b}

    common_seeds = sorted(set(seeds_a.keys()) & set(seeds_b.keys()))
    if not common_seeds:
        raise ValueError(f"No common seeds between {cond_a} and {cond_b}")

    # Get probe steps (should be same for all runs)
    first_run = runs_a[seeds_a[common_seeds[0]]]
    steps = np.array(first_run['probe_steps'])

    # Extract gaps for each seed
    gaps = []
    for seed in common_seeds:
        run_a = runs_a[seeds_a[seed]]
        run_b = runs_b[seeds_b[seed]]

        r2_a = np.array(run_a['probe_r2'][layer])
        r2_b = np.array(run_b['probe_r2'][layer])

        # Verify lengths match
        if len(r2_a) != len(steps) or len(r2_b) != len(steps):
            print(f"Warning: Mismatched lengths for seed {seed}, skipping")
            continue

        gaps.append(r2_a - r2_b)

    return steps, np.array(gaps), common_seeds


# =============================================================================
# Model Fitting
# =============================================================================

def fit_double_exp(steps, gap, n_restarts=10):
    """Fit double exponential: gap(t) = gap_inf + A1*exp(-t/tau1) + A2*exp(-t/tau2).

    Constraints: A1 > 0, A2 > 0, tau1 < tau2 (for identifiability)

    Returns:
        dict with gap_inf, params, rmse, bic, success
    """
    steps = np.array(steps, dtype=float)
    gap = np.array(gap, dtype=float)
    n = len(gap)

    def model(t, gap_inf, A1, A2, tau1, tau2):
        return gap_inf + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

    best_result = None
    best_rmse = np.inf

    # Grid of initial guesses
    gap_inf_guesses = [gap[-1], gap[-3:].mean(), gap[-5:].mean()]
    total_decay = gap[0] - gap[-1]

    for gap_inf_init in gap_inf_guesses:
        for A_ratio in [0.3, 0.5, 0.7]:
            for tau_ratio in [0.2, 0.33, 0.5]:
                A1_init = total_decay * A_ratio
                A2_init = total_decay * (1 - A_ratio)
                tau1_init = steps[-1] * tau_ratio * 0.5
                tau2_init = steps[-1] * tau_ratio * 1.5

                p0 = [gap_inf_init, max(A1_init, 0.001), max(A2_init, 0.001),
                      tau1_init, tau2_init]

                # Bounds: gap_inf can be anything, A1/A2 > 0, tau1 < tau2
                bounds = (
                    [-0.5, 0.001, 0.001, 100, 500],
                    [0.5, 1.0, 1.0, steps[-1], steps[-1] * 3]
                )

                try:
                    popt, _ = curve_fit(model, steps, gap, p0=p0, bounds=bounds,
                                       maxfev=1000)
                    pred = model(steps, *popt)
                    rmse = np.sqrt(np.mean((gap - pred)**2))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_result = popt
                except:
                    continue

    if best_result is None:
        return {'success': False}

    gap_inf, A1, A2, tau1, tau2 = best_result
    pred = model(steps, *best_result)
    ss_res = np.sum((gap - pred)**2)

    # BIC
    k = 5  # number of parameters
    sigma2 = ss_res / n
    log_lik = -n/2 * np.log(2*np.pi*sigma2) - n/2
    bic = k * np.log(n) - 2 * log_lik

    return {
        'success': True,
        'gap_inf': gap_inf,
        'params': {'A1': A1, 'A2': A2, 'tau1': tau1, 'tau2': tau2},
        'rmse': best_rmse,
        'bic': bic,
        'model_func': lambda t: model(t, *best_result),
        'name': 'Double exponential'
    }


def fit_single_exp(steps, gap):
    """Fit single exponential: gap(t) = gap_inf + A*exp(-t/tau).

    Returns:
        dict with gap_inf, params, rmse, bic, success
    """
    steps = np.array(steps, dtype=float)
    gap = np.array(gap, dtype=float)
    n = len(gap)

    def model(t, gap_inf, A, tau):
        return gap_inf + A * np.exp(-t / tau)

    # Initial guess
    gap_inf_init = gap[-1]
    A_init = gap[0] - gap[-1]
    tau_init = steps[-1] / 3

    p0 = [gap_inf_init, max(A_init, 0.001), tau_init]
    bounds = ([-0.5, 0.001, 100], [0.5, 1.0, steps[-1] * 3])

    try:
        popt, _ = curve_fit(model, steps, gap, p0=p0, bounds=bounds, maxfev=1000)
        gap_inf, A, tau = popt
        pred = model(steps, *popt)
        rmse = np.sqrt(np.mean((gap - pred)**2))
        ss_res = np.sum((gap - pred)**2)

        k = 3
        sigma2 = ss_res / n
        log_lik = -n/2 * np.log(2*np.pi*sigma2) - n/2
        bic = k * np.log(n) - 2 * log_lik

        return {
            'success': True,
            'gap_inf': gap_inf,
            'params': {'A': A, 'tau': tau},
            'rmse': rmse,
            'bic': bic,
            'model_func': lambda t: model(t, *popt),
            'name': 'Single exponential'
        }
    except:
        return {'success': False}


def fit_power_law(steps, gap):
    """Fit power law: gap(t) = gap_inf + A*(t+t0)^(-alpha).

    Returns:
        dict with gap_inf, params, rmse, bic, success
    """
    steps = np.array(steps, dtype=float)
    gap = np.array(gap, dtype=float)
    n = len(gap)

    def model(t, gap_inf, A, alpha, t0):
        return gap_inf + A * (t + t0)**(-alpha)

    # Initial guess
    gap_inf_init = gap[-1]
    A_init = (gap[0] - gap[-1]) * (steps[0] + 1000)**0.5

    p0 = [gap_inf_init, max(A_init, 0.1), 0.5, 1000]
    bounds = ([-0.5, 0.01, 0.1, 100], [0.5, 100, 2.0, steps[-1]])

    try:
        popt, _ = curve_fit(model, steps, gap, p0=p0, bounds=bounds, maxfev=1000)
        gap_inf, A, alpha, t0 = popt
        pred = model(steps, *popt)
        rmse = np.sqrt(np.mean((gap - pred)**2))
        ss_res = np.sum((gap - pred)**2)

        k = 4
        sigma2 = ss_res / n
        log_lik = -n/2 * np.log(2*np.pi*sigma2) - n/2
        bic = k * np.log(n) - 2 * log_lik

        return {
            'success': True,
            'gap_inf': gap_inf,
            'params': {'A': A, 'alpha': alpha, 't0': t0},
            'rmse': rmse,
            'bic': bic,
            'model_func': lambda t: model(t, *popt),
            'name': 'Power law'
        }
    except:
        return {'success': False}


def fit_double_power_law(steps, gap, n_restarts=10):
    """Fit double power law: gap(t) = gap_inf + A1*(t+t0)^(-alpha1) + A2*(t+t0)^(-alpha2).

    Uses shared t0 for identifiability. Constraint: alpha1 < alpha2.

    Parameters: 6 (gap_inf, A1, A2, alpha1, alpha2, t0)

    Returns:
        dict with gap_inf, params, rmse, bic, success, model_func, name
    """
    steps = np.array(steps, dtype=float)
    gap = np.array(gap, dtype=float)
    n = len(gap)

    def model(t, gap_inf, A1, A2, alpha1, alpha2, t0):
        return gap_inf + A1 * (t + t0)**(-alpha1) + A2 * (t + t0)**(-alpha2)

    best_result = None
    best_rmse = np.inf

    # Grid of initial guesses
    gap_inf_guesses = [gap[-1], gap[-3:].mean(), gap[-5:].mean()]
    total_decay = gap[0] - gap[-1]

    for gap_inf_init in gap_inf_guesses:
        for A_ratio in [0.3, 0.5, 0.7]:
            for alpha_pair in [(0.3, 0.8), (0.5, 1.2), (0.2, 0.6)]:
                A1_init = max(total_decay * A_ratio, 0.01)
                A2_init = max(total_decay * (1 - A_ratio), 0.01)
                alpha1_init, alpha2_init = alpha_pair
                t0_init = 1000

                p0 = [gap_inf_init, A1_init, A2_init,
                      alpha1_init, alpha2_init, t0_init]

                # Bounds: gap_inf flexible, A > 0, alpha in (0.1, 2), t0 positive
                bounds = (
                    [-0.5, 0.001, 0.001, 0.1, 0.1, 100],
                    [0.5, 10, 10, 2.0, 2.0, steps[-1] * 2]
                )

                try:
                    popt, _ = curve_fit(model, steps, gap, p0=p0, bounds=bounds, maxfev=1000)

                    # Enforce alpha1 < alpha2 constraint
                    if popt[3] >= popt[4]:
                        continue

                    pred = model(steps, *popt)
                    rmse = np.sqrt(np.mean((gap - pred)**2))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_result = popt
                except:
                    continue

    if best_result is None:
        return {'success': False}

    gap_inf, A1, A2, alpha1, alpha2, t0 = best_result
    pred = model(steps, *best_result)
    ss_res = np.sum((gap - pred)**2)

    k = 6  # number of parameters
    sigma2 = ss_res / n
    log_lik = -n/2 * np.log(2*np.pi*sigma2) - n/2
    bic = k * np.log(n) - 2 * log_lik

    # Capture best_result in closure
    final_params = best_result.copy()

    return {
        'success': True,
        'gap_inf': gap_inf,
        'params': {'A1': A1, 'A2': A2, 'alpha1': alpha1, 'alpha2': alpha2, 't0': t0},
        'rmse': best_rmse,
        'bic': bic,
        'model_func': lambda t, p=final_params: model(t, *p),
        'name': 'Double power law'
    }


def fit_all_models(steps, gap):
    """Fit all 4 models and return results dict.

    Returns:
        dict mapping model_name -> result dict (including failed models with success=False)
    """
    fit_funcs = {
        'Double exponential': fit_double_exp,
        'Single exponential': fit_single_exp,
        'Power law': fit_power_law,
    }

    results = {}
    for name, fit_func in fit_funcs.items():
        results[name] = fit_func(steps, gap)
        if results[name]['success']:
            results[name]['name'] = name

    return results


def select_best_model(steps, gap):
    """Fit all models and return the best one by BIC.

    Returns:
        Best model dict, or None if all fail
    """
    all_results = fit_all_models(steps, gap)
    successful = [r for r in all_results.values() if r['success']]

    if not successful:
        return None

    # Select by BIC (lower is better)
    return min(successful, key=lambda r: r['bic'])


# =============================================================================
# Bootstrap
# =============================================================================

def bootstrap_gap_inf(steps, gaps_all_runs, n_bootstrap=1000, verbose=True):
    """Bootstrap confidence interval for asymptotic gap using single exponential.

    Procedure:
    1. Fit single exponential to mean trajectory for point estimate
    2. For each bootstrap iteration:
       - For each probe step, randomly choose one of the runs
       - Build virtual trajectory from these choices
       - Fit single exponential, extract gap_inf
    3. Compute CI from distribution of gap_inf values

    Args:
        steps: (n_steps,) array of probe steps
        gaps_all_runs: (n_runs, n_steps) array of gap values
        n_bootstrap: number of bootstrap iterations
        verbose: print progress

    Returns:
        dict with gap_inf, ci_95, bootstrap_samples, etc.
    """
    steps = np.array(steps, dtype=float)
    gaps_all_runs = np.array(gaps_all_runs)
    n_runs, n_steps = gaps_all_runs.shape

    # Step 1: Fit single exponential to mean trajectory
    mean_gap = np.mean(gaps_all_runs, axis=0)
    point_fit = fit_single_exp(steps, mean_gap)

    if not point_fit['success']:
        print("Warning: Could not fit single exponential to mean trajectory")
        return {'success': False}

    if verbose:
        print(f"  Single exponential fit: gap_inf={point_fit['gap_inf']:.4f}, RMSE={point_fit['rmse']:.5f}")

    # Step 2: Bootstrap using single exponential
    bootstrap_samples = []
    n_failures = 0
    start_time = time.time()

    # Calculate progress report interval (every 10% or at least every 100 iterations)
    report_interval = max(100, n_bootstrap // 10)

    if verbose:
        print(f"  Running {n_bootstrap} bootstrap iterations...")

    for b in range(n_bootstrap):
        if verbose and (b + 1) % report_interval == 0:
            elapsed = time.time() - start_time
            rate = (b + 1) / elapsed
            eta = (n_bootstrap - b - 1) / rate
            print(f"    {b + 1}/{n_bootstrap} ({elapsed:.1f}s elapsed, {eta:.1f}s remaining)")

        # For each step, randomly choose one run
        run_choices = np.random.randint(0, n_runs, size=n_steps)
        virtual_gap = gaps_all_runs[run_choices, np.arange(n_steps)]

        # Fit single exponential
        result = fit_single_exp(steps, virtual_gap)

        if result['success']:
            bootstrap_samples.append(result['gap_inf'])
        else:
            # Fallback: use last few values as estimate
            bootstrap_samples.append(np.mean(virtual_gap[-5:]))
            n_failures += 1

    if verbose and n_failures > 0:
        print(f"  Warning: {n_failures}/{n_bootstrap} fits failed, used fallback")

    bootstrap_samples = np.array(bootstrap_samples)

    # Step 3: Compute statistics
    ci_95 = (np.percentile(bootstrap_samples, 2.5),
             np.percentile(bootstrap_samples, 97.5))
    ci_90 = (np.percentile(bootstrap_samples, 5),
             np.percentile(bootstrap_samples, 95))

    # Fraction positive
    frac_positive = np.mean(bootstrap_samples > 0)

    return {
        'success': True,
        'gap_inf': point_fit['gap_inf'],
        'ci_95': ci_95,
        'ci_90': ci_90,
        'se': np.std(bootstrap_samples),
        'bootstrap_samples': bootstrap_samples,
        'frac_positive': frac_positive,
        'point_fit': point_fit,
        'n_fit_failures': n_failures
    }


def bootstrap_exp_and_power(steps, gaps_all_runs, n_bootstrap=5000, verbose=True):
    """Bootstrap exponential and power law models to compare asymptotic estimates.

    Bootstraps the two main competing models (exponential vs power law) to
    check if they agree on gap_inf or show model uncertainty.

    Returns:
        dict with:
            - per_model: {model_name: {gap_inf, ci_95, se, bootstrap_samples, n_failures}}
            - agreement: bool (True if 95% CIs overlap)
            - bic_selected: str (name of BIC-best model)
            - success: bool
    """
    steps = np.array(steps, dtype=float)
    gaps_all_runs = np.array(gaps_all_runs)
    n_runs, n_steps = gaps_all_runs.shape

    # Fit both models to mean trajectory for BIC comparison
    mean_gap = np.mean(gaps_all_runs, axis=0)

    exp_fit = fit_single_exp(steps, mean_gap)
    power_fit = fit_power_law(steps, mean_gap)

    if not exp_fit['success'] or not power_fit['success']:
        return {'success': False}

    # Select BIC-best
    models = {
        'Single exponential': exp_fit,
        'Power law': power_fit
    }
    bic_selected = min(models.keys(), key=lambda k: models[k]['bic'])

    if verbose:
        print(f"  BIC-selected model: {bic_selected}")
        for name, fit in models.items():
            marker = " <--" if name == bic_selected else ""
            print(f"    {name}: BIC={fit['bic']:.1f}, RMSE={fit['rmse']:.5f}{marker}")

    # Bootstrap each model
    per_model_results = {}

    # Calculate progress report interval (every 10% or at least every 100 iterations)
    report_interval = max(100, n_bootstrap // 10)

    for model_idx, (model_name, fit_func) in enumerate([('Single exponential', fit_single_exp),
                                                          ('Power law', fit_power_law)]):
        if verbose:
            print(f"  [{model_idx + 1}/2] Bootstrapping {model_name}...")

        bootstrap_samples = []
        n_failures = 0
        model_start = time.time()

        for b in range(n_bootstrap):
            if verbose and (b + 1) % report_interval == 0:
                elapsed = time.time() - model_start
                rate = (b + 1) / elapsed
                eta = (n_bootstrap - b - 1) / rate
                print(f"    {b + 1}/{n_bootstrap} ({elapsed:.1f}s elapsed, {eta:.1f}s remaining)")

            run_choices = np.random.randint(0, n_runs, size=n_steps)
            virtual_gap = gaps_all_runs[run_choices, np.arange(n_steps)]

            result = fit_func(steps, virtual_gap)
            if result['success']:
                bootstrap_samples.append(result['gap_inf'])
            else:
                bootstrap_samples.append(np.mean(virtual_gap[-5:]))
                n_failures += 1

        bootstrap_samples = np.array(bootstrap_samples)
        ci_95 = (np.percentile(bootstrap_samples, 2.5), np.percentile(bootstrap_samples, 97.5))

        per_model_results[model_name] = {
            'gap_inf': models[model_name]['gap_inf'],
            'ci_95': ci_95,
            'se': np.std(bootstrap_samples),
            'bootstrap_samples': bootstrap_samples,
            'frac_positive': np.mean(bootstrap_samples > 0),
            'n_failures': n_failures,
            'bic': models[model_name]['bic'],
        }

    # Check agreement
    cis = [r['ci_95'] for r in per_model_results.values()]
    max_low = max(ci[0] for ci in cis)
    min_high = min(ci[1] for ci in cis)
    agreement = max_low <= min_high

    return {
        'success': True,
        'per_model': per_model_results,
        'bic_selected': bic_selected,
        'agreement': agreement,
    }


def bootstrap_all_models(steps, gaps_all_runs, n_bootstrap=5000, verbose=True):
    """Bootstrap ALL models and check agreement on gap_inf.

    Instead of selecting one model by BIC, this bootstraps all 4 models
    independently to check if they agree on the asymptotic gap.

    Returns:
        dict with:
            - per_model: {model_name: {gap_inf, ci_95, se, bootstrap_samples, n_failures}}
            - agreement: bool (True if all 95% CIs overlap)
            - bic_selected: str (name of BIC-best model)
            - combined_ci: (low, high) if models agree, else None
    """
    steps = np.array(steps, dtype=float)
    gaps_all_runs = np.array(gaps_all_runs)
    n_runs, n_steps = gaps_all_runs.shape

    # Fit all models to mean trajectory for BIC comparison
    mean_gap = np.mean(gaps_all_runs, axis=0)
    all_fits = fit_all_models(steps, mean_gap)

    successful_models = {k: v for k, v in all_fits.items() if v['success']}
    if not successful_models:
        return {'success': False}

    # Select BIC-best
    bic_selected = min(successful_models.keys(), key=lambda k: successful_models[k]['bic'])
    if verbose:
        print(f"  BIC-selected model: {bic_selected}")
        for name, fit in successful_models.items():
            marker = " <--" if name == bic_selected else ""
            print(f"    {name}: BIC={fit['bic']:.1f}, RMSE={fit['rmse']:.5f}{marker}")

    # Define fit functions
    fit_func_map = {
        'Double exponential': fit_double_exp,
        'Single exponential': fit_single_exp,
        'Power law': fit_power_law,
    }

    # Bootstrap each successful model
    per_model_results = {}

    # Calculate progress report interval (every 10% or at least every 100 iterations)
    report_interval = max(100, n_bootstrap // 10)

    for model_idx, model_name in enumerate(successful_models):
        if verbose:
            print(f"  [{model_idx + 1}/{len(successful_models)}] Bootstrapping {model_name}...")

        fit_func = fit_func_map[model_name]
        bootstrap_samples = []
        n_failures = 0
        model_start = time.time()

        for b in range(n_bootstrap):
            if verbose and (b + 1) % report_interval == 0:
                elapsed = time.time() - model_start
                rate = (b + 1) / elapsed
                eta = (n_bootstrap - b - 1) / rate
                print(f"    {b + 1}/{n_bootstrap} ({elapsed:.1f}s elapsed, {eta:.1f}s remaining)")

            run_choices = np.random.randint(0, n_runs, size=n_steps)
            virtual_gap = gaps_all_runs[run_choices, np.arange(n_steps)]

            result = fit_func(steps, virtual_gap)
            if result['success']:
                bootstrap_samples.append(result['gap_inf'])
            else:
                bootstrap_samples.append(np.mean(virtual_gap[-5:]))
                n_failures += 1

        bootstrap_samples = np.array(bootstrap_samples)
        ci_95 = (np.percentile(bootstrap_samples, 2.5), np.percentile(bootstrap_samples, 97.5))

        per_model_results[model_name] = {
            'gap_inf': successful_models[model_name]['gap_inf'],
            'ci_95': ci_95,
            'se': np.std(bootstrap_samples),
            'bootstrap_samples': bootstrap_samples,
            'frac_positive': np.mean(bootstrap_samples > 0),
            'n_failures': n_failures,
            'bic': successful_models[model_name]['bic'],
            'rmse': successful_models[model_name]['rmse'],
        }

        if verbose:
            model_elapsed = time.time() - model_start
            print(f"    Done in {model_elapsed:.1f}s" + (f" ({n_failures} fits failed)" if n_failures > 0 else ""))

    # Check agreement: do all 95% CIs overlap?
    intervals = [per_model_results[k]['ci_95'] for k in per_model_results]
    max_lower = max(i[0] for i in intervals)
    min_upper = min(i[1] for i in intervals)
    agreement = max_lower <= min_upper

    combined_ci = (max_lower, min_upper) if agreement else None

    if verbose:
        print(f"\n  Model agreement: {'YES' if agreement else 'NO'}")
        if agreement:
            print(f"  Combined CI (intersection): [{combined_ci[0]:.4f}, {combined_ci[1]:.4f}]")

    return {
        'success': True,
        'per_model': per_model_results,
        'agreement': agreement,
        'bic_selected': bic_selected,
        'combined_ci': combined_ci,
        'point_fits': all_fits,  # For plotting all model curves
    }


# =============================================================================
# Alpha Sweep Analysis
# =============================================================================

def run_alpha_sweep(checkpoint_path, data_path='data/combined_for_experiment.npz',
                    alphas=None, n_samples=2000, device=None, verbose=True):
    """Re-run probing with multiple alpha values for a model checkpoint.

    Args:
        checkpoint_path: Path to model .pt file
        data_path: Path to game data .npz
        alphas: Array of alpha values (default: logspace(-6, 6, 100))
        n_samples: Number of samples for probing
        device: Compute device (default: auto-detect)

    Returns:
        dict with:
            - alphas: array of alpha values tested
            - layer_r2: {str(layer_idx): array of R² values}
            - optimal_alpha: {str(layer_idx): best alpha value}
            - sensitivity: {str(layer_idx): std of R² across alphas}
    """
    from sklearn.linear_model import Ridge

    model_mod = _import_model()
    data_loader_mod = _import_data_loader()
    probing_mod = _import_probing()
    training_mod = _import_training()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if alphas is None:
        alphas = np.logspace(-6, 6, 100)

    if verbose:
        print(f"  Loading checkpoint: {checkpoint_path}")

    # Determine model type from checkpoint name
    checkpoint_name = Path(checkpoint_path).stem.lower()
    if 'large' in checkpoint_name:
        model = model_mod.create_large_model()
    else:
        model = model_mod.create_small_model()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    if verbose:
        print(f"  Loading data from {data_path}...")

    # Load data
    dataset = data_loader_mod.EfficientDomineeringDataset(data_path, split='val', positions_per_game=20)
    dataset.precompute_epoch()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=200, shuffle=False,
        num_workers=0, collate_fn=training_mod.collate_batch
    )

    if verbose:
        print(f"  Collecting activations from {n_samples} samples...")

    # Collect activations
    all_activations = {}
    all_sectors = []
    samples_collected = 0

    with torch.no_grad():
        for batch in loader:
            if samples_collected >= n_samples:
                break
            tokens = batch['tokens'].to(device)
            layer_acts = probing_mod.get_all_layer_activations(model, tokens)
            for layer_idx, acts in layer_acts.items():
                if layer_idx not in all_activations:
                    all_activations[layer_idx] = []
                all_activations[layer_idx].append(acts.cpu().numpy())
            all_sectors.append(batch['sectors'].numpy())
            samples_collected += len(tokens)

    # Concatenate
    for layer_idx in all_activations:
        all_activations[layer_idx] = np.concatenate(all_activations[layer_idx], axis=0)[:n_samples]
    all_sectors = np.concatenate(all_sectors, axis=0)[:n_samples]

    # Train/val split
    n_total = len(all_sectors)
    indices = np.random.permutation(n_total)
    n_val = int(0.2 * n_total)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    y_train, y_val = all_sectors[train_idx], all_sectors[val_idx]

    if verbose:
        print(f"  Testing {len(alphas)} alpha values across {len(all_activations)} layers...")

    # Sweep alphas for each layer
    layer_r2 = {}
    optimal_alpha = {}

    for layer_idx in sorted(all_activations.keys(), key=lambda x: (x < 0, x)):
        X = all_activations[layer_idx]
        X_train, X_val = X[train_idx], X[val_idx]

        r2_values = []
        for alpha in alphas:
            probe = Ridge(alpha=alpha)
            probe.fit(X_train, y_train)
            r2 = probe.score(X_val, y_val)
            r2_values.append(r2)

        r2_values = np.array(r2_values)
        layer_r2[str(layer_idx)] = r2_values
        optimal_alpha[str(layer_idx)] = alphas[np.argmax(r2_values)]

        if verbose:
            best_r2 = np.max(r2_values)
            best_alpha = optimal_alpha[str(layer_idx)]
            print(f"    Layer {layer_idx}: best R²={best_r2:.4f} at alpha={best_alpha:.2e}")

    # Compute sensitivity (std of R² across alphas)
    sensitivity = {k: np.std(v) for k, v in layer_r2.items()}

    return {
        'alphas': alphas,
        'layer_r2': layer_r2,
        'optimal_alpha': optimal_alpha,
        'sensitivity': sensitivity,
        'checkpoint': str(checkpoint_path),
    }


def plot_alpha_sweep(results, output_path, title="Alpha Sweep"):
    """Plot R² vs alpha curves for each layer."""
    alphas = results['alphas']
    layer_r2 = results['layer_r2']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map for layers
    n_layers = len(layer_r2)
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

    for i, (layer_str, r2_values) in enumerate(sorted(layer_r2.items(), key=lambda x: (int(x[0]) < 0, int(x[0])))):
        layer_label = f"Layer {layer_str}" if layer_str != '-1' else "Post-LN"
        ax.semilogx(alphas, r2_values, '-', color=colors[i], linewidth=2, label=layer_label)

        # Mark optimal
        opt_alpha = results['optimal_alpha'][layer_str]
        opt_r2 = np.max(r2_values)
        ax.scatter([opt_alpha], [opt_r2], color=colors[i], s=50, zorder=10)

    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='alpha=1.0 (default)')

    ax.set_xlabel('Ridge Alpha (log scale)', fontsize=11)
    ax.set_ylabel('Probe R²', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved alpha sweep plot to {output_path}")


def run_alpha_sweep_all_models(checkpoints_dir='checkpoints', data_path='data/combined_for_experiment.npz',
                                output_dir='analysis_results/alpha_sweep', alphas=None,
                                n_samples=2000, device=None, verbose=True):
    """Run alpha sweep on all large model checkpoints.

    Returns:
        dict mapping model_name -> alpha_sweep_results
    """
    checkpoints_dir = Path(checkpoints_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all large model checkpoints
    large_checkpoints = list(checkpoints_dir.glob('large_*_best.pt'))
    if not large_checkpoints:
        print(f"No large model checkpoints found in {checkpoints_dir}")
        return {}

    if verbose:
        print(f"Found {len(large_checkpoints)} large model checkpoints")

    all_results = {}
    for ckpt_path in sorted(large_checkpoints):
        model_name = ckpt_path.stem.replace('_best', '')
        if verbose:
            print(f"\nProcessing {model_name}...")

        results = run_alpha_sweep(ckpt_path, data_path, alphas, n_samples, device, verbose)
        all_results[model_name] = results

        # Plot individual
        plot_path = output_dir / f"alpha_sweep_{model_name}.png"
        plot_alpha_sweep(results, plot_path, title=f"Alpha Sweep: {model_name}")

    # Save results JSON
    results_json = output_dir / 'alpha_sweep_results.json'
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for model_name, res in all_results.items():
        json_results[model_name] = {
            'alphas': res['alphas'].tolist(),
            'layer_r2': {k: v.tolist() for k, v in res['layer_r2'].items()},
            'optimal_alpha': {k: float(v) for k, v in res['optimal_alpha'].items()},
            'sensitivity': {k: float(v) for k, v in res['sensitivity'].items()},
            'checkpoint': res['checkpoint'],
        }
    with open(results_json, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved all alpha sweep results to {results_json}")

    return all_results


# =============================================================================
# Permutation Test
# =============================================================================

def permutation_test_sectors(checkpoint_path, data_path='data/combined_for_experiment.npz',
                             n_permutations=1000, layer=2, alpha=1.0,
                             n_samples=2000, device=None, verbose=True):
    """Test if probe learned the specific spatial sector mapping.

    Trains a probe on correct sector labels, then tests whether R² drops
    when sector columns are permuted at evaluation time. This tests whether
    the probe learned which 4x4 region corresponds to which output.

    If R² drops significantly under permutation, the probe learned the
    specific spatial structure, not just that sectors are predictable.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to game data
        n_permutations: Number of permutation samples
        layer: Which layer to test (default: 2 = embedded layer)
        alpha: Ridge alpha
        n_samples: Number of samples for probing
        device: Compute device

    Returns:
        dict with:
            - true_r2: R² with correct sector mapping
            - null_r2_mean: Mean R² under shuffled mappings
            - null_r2_std: Std of null distribution
            - p_value: Fraction of null R² >= true_r2
            - null_distribution: Array of null R² values
    """
    from sklearn.linear_model import Ridge

    model_mod = _import_model()
    data_loader_mod = _import_data_loader()
    probing_mod = _import_probing()
    training_mod = _import_training()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print(f"  Loading checkpoint: {checkpoint_path}")

    # Load model
    checkpoint_name = Path(checkpoint_path).stem.lower()
    if 'large' in checkpoint_name:
        model = model_mod.create_large_model()
    else:
        model = model_mod.create_small_model()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    if verbose:
        print(f"  Loading data...")

    # Load data
    dataset = data_loader_mod.EfficientDomineeringDataset(data_path, split='val', positions_per_game=20)
    dataset.precompute_epoch()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=200, shuffle=False,
        num_workers=0, collate_fn=training_mod.collate_batch
    )

    if verbose:
        print(f"  Collecting activations for layer {layer}...")

    # Collect activations and targets
    all_activations = []
    all_sectors = []
    samples_collected = 0

    with torch.no_grad():
        for batch in loader:
            if samples_collected >= n_samples:
                break
            tokens = batch['tokens'].to(device)
            acts = probing_mod.get_layer_activations(model, tokens, layer)
            all_activations.append(acts.cpu().numpy())
            all_sectors.append(batch['sectors'].numpy())
            samples_collected += len(tokens)

    X = np.concatenate(all_activations, axis=0)[:n_samples]
    Y = np.concatenate(all_sectors, axis=0)[:n_samples]  # (n_samples, 16)

    # Split
    n = len(X)
    n_val = int(0.2 * n)
    indices = np.random.permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # Train probe once on correct labels
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, Y_train)
    true_r2 = probe.score(X_val, Y_val)

    if verbose:
        print(f"  True R²: {true_r2:.4f}")
        print(f"  Running {n_permutations} permutations...")

    # Permutation test: train on correct, test on permuted
    # This tests whether the probe learned the specific spatial mapping
    null_r2 = []
    for p in range(n_permutations):
        if verbose and (p + 1) % 200 == 0:
            print(f"    {p + 1}/{n_permutations}")

        # Permute the 16 sector columns at test time only
        perm = np.random.permutation(16)
        Y_val_perm = Y_val[:, perm]

        # Evaluate the SAME probe (trained on correct labels) on permuted targets
        r2 = probe.score(X_val, Y_val_perm)
        null_r2.append(r2)

    null_r2 = np.array(null_r2)
    # p-value: how often does permuted R² exceed true R²?
    # If probe learned specific mapping, permuted R² should be lower
    p_value = np.mean(null_r2 >= true_r2)

    if verbose:
        print(f"\n  Results:")
        print(f"    True R²: {true_r2:.4f}")
        print(f"    Permuted R² mean: {np.mean(null_r2):.4f} (std: {np.std(null_r2):.4f})")
        print(f"    p-value: {p_value:.4f}")
        if p_value < 0.01:
            print(f"    --> SIGNIFICANT: Probe learned specific spatial mapping (p < 0.01)")
        else:
            print(f"    --> NOT significant: Probe may not encode spatial structure")

    return {
        'true_r2': true_r2,
        'null_r2_mean': np.mean(null_r2),
        'null_r2_std': np.std(null_r2),
        'p_value': p_value,
        'null_distribution': null_r2,
        'layer': layer,
        'alpha': alpha,
        'checkpoint': str(checkpoint_path),
    }


def plot_permutation_test(result, output_path):
    """Plot permutation test results."""
    fig, ax = plt.subplots(figsize=(8, 5))

    null_dist = result['null_distribution']
    true_r2 = result['true_r2']

    ax.hist(null_dist, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', label=f'Permuted R² (n={len(null_dist)})')

    ax.axvline(true_r2, color='red', linewidth=2, linestyle='-',
               label=f'True R² = {true_r2:.4f}')
    ax.axvline(np.mean(null_dist), color='orange', linewidth=2, linestyle='--',
               label=f'Permuted mean = {np.mean(null_dist):.4f}')

    # p-value annotation
    p_val = result['p_value']
    if p_val < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p_val:.3f}"

    sig_text = "Spatial mapping learned" if p_val < 0.01 else "No spatial specificity"
    color = 'green' if p_val < 0.01 else 'orange'

    ax.text(0.95, 0.95, f"{p_text}\n{sig_text}", transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    ax.set_xlabel('Probe R² (permuted sectors)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Spatial Mapping Test (Layer {result["layer"]})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved permutation test plot to {output_path}")


# =============================================================================
# Layer Comparison Analysis
# =============================================================================

def analyze_layer_comparison(histories):
    """Compare probe R² across layers for each model type.

    Tests the hypothesis that embedded+aux models concentrate
    sector information in layer 2.

    Returns:
        dict with layer profiles and prominence scores
    """
    model_types = ['Large-baseline', 'Large+embed(aux)', 'Large+embed(noaux)']

    # Group histories by model type
    type_histories = {t: [] for t in model_types}
    for name, hist in histories.items():
        for model_type in model_types:
            if model_type in name:
                type_histories[model_type].append(hist)
                break

    results = {
        'layer_profiles': {},
        'peak_layer': {},
        'layer_2_prominence': {},
        'seed_data': {},
    }

    for model_type, hists in type_histories.items():
        if not hists:
            continue

        # Collect final R² for each layer across seeds
        layer_r2_by_seed = {}
        for hist in hists:
            probe_r2 = hist.get('probe_r2', {})
            for layer_str, r2_list in probe_r2.items():
                layer = int(layer_str) if layer_str != '-1' else -1
                if layer not in layer_r2_by_seed:
                    layer_r2_by_seed[layer] = []
                if r2_list:  # Take final value
                    layer_r2_by_seed[layer].append(r2_list[-1])

        # Compute mean profile
        layer_profile = {l: np.mean(r2s) for l, r2s in layer_r2_by_seed.items()}
        results['layer_profiles'][model_type] = layer_profile
        results['seed_data'][model_type] = {l: r2s for l, r2s in layer_r2_by_seed.items()}

        # Peak layer
        if layer_profile:
            results['peak_layer'][model_type] = max(layer_profile.keys(),
                                                    key=lambda l: layer_profile[l])

        # Layer 2 prominence (specific to embedding hypothesis)
        if 2 in layer_profile:
            other_layers = [v for l, v in layer_profile.items() if l != 2]
            if other_layers:
                results['layer_2_prominence'][model_type] = layer_profile[2] / np.mean(other_layers)

    return results


def plot_layer_comparison(results, output_path):
    """Plot layer comparison results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_colors = {
        'Large-baseline': '#1f77b4',
        'Large+embed(aux)': '#2ca02c',
        'Large+embed(noaux)': '#ff7f0e',
    }

    # Panel 1: Baseline layer profile (bar chart)
    ax = axes[0]
    baseline_profile = results['layer_profiles'].get('Large-baseline', {})
    baseline_seed_data = results['seed_data'].get('Large-baseline', {})

    layers = sorted(baseline_profile.keys(), key=lambda x: (x < 0, x))
    r2_values = [baseline_profile[l] for l in layers]
    stds = [np.std(baseline_seed_data.get(l, [0])) for l in layers]
    x_labels = [str(l) if l >= 0 else 'post-LN' for l in layers]

    bars = ax.bar(range(len(layers)), r2_values, color='#1f77b4', alpha=0.7,
                  yerr=stds, capsize=4, error_kw={'linewidth': 1.5})

    # Highlight layer 2 (embedded layer)
    if 2 in layers:
        layer_2_idx = layers.index(2)
        bars[layer_2_idx].set_color('#d62728')
        bars[layer_2_idx].set_alpha(0.8)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Final Probe R²', fontsize=11)
    ax.set_title('Baseline Model: R² by Layer', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Layer 2 prominence
    ax = axes[1]
    model_types = list(results['layer_2_prominence'].keys())
    prominence_values = [results['layer_2_prominence'][m] for m in model_types]
    colors = [model_colors.get(m, 'gray') for m in model_types]

    bars = ax.bar(range(len(model_types)), prominence_values, color=colors, alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No concentration (ratio=1)')

    ax.set_xticks(range(len(model_types)))
    ax.set_xticklabels([m.replace('Large', '').replace('+', '\n+').replace('-', '\n-')
                        for m in model_types], fontsize=9)
    ax.set_ylabel('Layer 2 Prominence\n(R²_layer2 / mean_other)', fontsize=11)
    ax.set_title('Sector Info Concentration in Layer 2', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved layer comparison plot to {output_path}")


# =============================================================================
# Diagnostic Tests
# =============================================================================

def durbin_watson_test(residuals):
    """Compute Durbin-Watson statistic for residual autocorrelation.

    DW range: 0-4
    - DW ~ 2: No autocorrelation (good)
    - DW < 1.5: Positive autocorrelation (model may miss systematic structure)
    - DW > 2.5: Negative autocorrelation (rare)
    """
    residuals = np.array(residuals)
    dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)

    if dw < 1.5:
        interpretation = "Positive autocorrelation - model may miss systematic structure"
    elif dw > 2.5:
        interpretation = "Negative autocorrelation - unusual, check data"
    else:
        interpretation = "Acceptable - no significant autocorrelation"

    return {'dw': dw, 'interpretation': interpretation}


def cross_seed_consistency(values, metric_name="metric"):
    """Compute consistency metrics across seeds.

    Returns coefficient of variation (CV) - lower is more consistent.
    CV < 10% indicates robust results.
    """
    values = np.array(values)
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv = (std_val / abs(mean_val) * 100) if mean_val != 0 else float('inf')

    return {
        'mean': mean_val,
        'std': std_val,
        'cv_percent': cv,
        'range': (float(np.min(values)), float(np.max(values))),
        'n_seeds': len(values),
        'interpretation': "Robust" if cv < 10 else "Variable" if cv < 25 else "High variance"
    }


def check_bootstrap_convergence(samples, checkpoints=None):
    """Check if bootstrap CI has converged.

    Returns True if CI width changes < 5% in last 3 checkpoints.
    """
    if checkpoints is None:
        checkpoints = [500, 1000, 2000, 3000, 4000, 5000]

    samples = np.array(samples)
    ci_widths = []

    for n in checkpoints:
        if n > len(samples):
            break
        subsample = samples[:n]
        ci = (np.percentile(subsample, 2.5), np.percentile(subsample, 97.5))
        ci_widths.append(ci[1] - ci[0])

    if len(ci_widths) >= 3:
        recent_changes = [abs(ci_widths[i] - ci_widths[i-1]) / ci_widths[i-1]
                          for i in range(-2, 0)]
        converged = all(c < 0.05 for c in recent_changes)
    else:
        converged = False

    return {
        'converged': converged,
        'ci_widths': dict(zip(checkpoints[:len(ci_widths)], ci_widths)),
        'recommendation': "Sufficient samples" if converged else "Consider more samples"
    }


# =============================================================================
# Autocorrelation Tests for Bootstrap Validity
# =============================================================================

def compute_within_acf(gaps_all_runs, max_lag):
    """Compute within-seed autocorrelation function.

    For each seed, computes autocorrelation at lags 1..max_lag, then averages
    across seeds.

    Args:
        gaps_all_runs: (n_runs, n_steps) array of gap values
        max_lag: Maximum lag to compute

    Returns:
        (max_lag,) array of within-seed autocorrelations
    """
    n_runs, n_steps = gaps_all_runs.shape
    within_acf = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        acf_values = []
        for seed_idx in range(n_runs):
            series = gaps_all_runs[seed_idx, :]
            if len(series) > lag:
                # Pearson correlation between series[:-lag] and series[lag:]
                x = series[:-lag]
                y = series[lag:]
                if len(x) > 1 and np.std(x) > 1e-10 and np.std(y) > 1e-10:
                    acf = np.corrcoef(x, y)[0, 1]
                    if not np.isnan(acf):
                        acf_values.append(acf)

        within_acf[lag - 1] = np.mean(acf_values) if acf_values else 0.0

    return within_acf


def compute_between_ccf(gaps_all_runs, max_lag):
    """Compute between-seed cross-correlation function.

    For each pair of different seeds, computes cross-correlation at lags 1..max_lag,
    then averages across all pairs.

    Args:
        gaps_all_runs: (n_runs, n_steps) array of gap values
        max_lag: Maximum lag to compute

    Returns:
        (max_lag,) array of between-seed cross-correlations
    """
    n_runs, n_steps = gaps_all_runs.shape
    between_ccf = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        ccf_values = []
        for i in range(n_runs):
            for j in range(n_runs):
                if i != j:
                    series_i = gaps_all_runs[i, :]
                    series_j = gaps_all_runs[j, :]
                    if len(series_i) > lag:
                        # Cross-correlation: series_i[:-lag] vs series_j[lag:]
                        x = series_i[:-lag]
                        y = series_j[lag:]
                        if len(x) > 1 and np.std(x) > 1e-10 and np.std(y) > 1e-10:
                            ccf = np.corrcoef(x, y)[0, 1]
                            if not np.isnan(ccf):
                                ccf_values.append(ccf)

        between_ccf[lag - 1] = np.mean(ccf_values) if ccf_values else 0.0

    return between_ccf


def permutation_test_autocorr(gaps_all_runs, max_lag, n_permutations, observed_tr):
    """Generate null distribution by shuffling within each seed.

    Tests whether observed threat ratios are significantly higher than would
    be expected by chance. Shuffles time points within each seed independently
    to destroy temporal structure while preserving marginal distribution.

    Args:
        gaps_all_runs: (n_runs, n_steps) array of gap values
        max_lag: Maximum lag to test
        n_permutations: Number of permutation iterations
        observed_tr: (max_lag,) array of observed threat ratios

    Returns:
        p_values: (max_lag,) array of p-values (fraction of null >= observed)
        null_distribution: (n_permutations, max_lag) array of null threat ratios
    """
    n_runs, n_steps = gaps_all_runs.shape
    null_threat_ratios = np.zeros((n_permutations, max_lag))

    for perm_idx in range(n_permutations):
        # Shuffle each seed's trajectory independently
        shuffled = np.zeros_like(gaps_all_runs)
        for i in range(n_runs):
            perm = np.random.permutation(n_steps)
            shuffled[i, :] = gaps_all_runs[i, perm]

        # Compute ACF and CCF on shuffled data
        within_null = compute_within_acf(shuffled, max_lag)
        between_null = compute_between_ccf(shuffled, max_lag)

        # Compute threat ratios
        epsilon = 1e-6
        tr_null = within_null / (np.abs(between_null) + epsilon)
        tr_null = np.minimum(tr_null, 100.0)
        null_threat_ratios[perm_idx, :] = tr_null

    # Compute p-values for each lag
    p_values = np.mean(null_threat_ratios >= observed_tr[np.newaxis, :], axis=0)

    return p_values, null_threat_ratios


def compute_effective_n(acf_values, n_steps):
    """Compute effective sample size using Bartlett's formula.

    Formula: n_eff = n / (1 + 2 * sum(rho_k))
    where sum is over significant autocorrelations.

    Args:
        acf_values: Array of autocorrelation values at lags 1, 2, ...
        n_steps: Actual number of time points

    Returns:
        n_eff: Effective sample size
        vif: Variance inflation factor (n_steps / n_eff)
    """
    # Sum autocorrelations up to cutoff
    K = min(len(acf_values), n_steps // 4)
    acf_sum = 0.0

    for k, acf_k in enumerate(acf_values[:K]):
        if abs(acf_k) < 0.05:  # Insignificant autocorrelation
            break
        acf_sum += acf_k

    # Ensure denominator is positive
    denominator = max(1.0, 1 + 2 * acf_sum)
    n_eff = n_steps / denominator
    vif = n_steps / n_eff

    return n_eff, vif


def classify_threat(threat_ratio, p_values, vif, within_acf):
    """Classify overall threat to bootstrap validity.

    Uses multiple criteria for robustness:
    1. Threat ratio magnitude (within vs between strength)
    2. Statistical significance (permutation test)
    3. Variance inflation (effective sample size reduction)
    4. Absolute autocorrelation level

    Args:
        threat_ratio: (max_lag,) array of threat ratios
        p_values: (max_lag,) array of p-values
        vif: Variance inflation factor
        within_acf: (max_lag,) array of within-seed autocorrelations

    Returns:
        'low', 'moderate', or 'high'
    """
    # Focus on lag 1 (most critical for bootstrap)
    tr_lag1 = threat_ratio[0]
    p_lag1 = p_values[0]
    acf_lag1 = within_acf[0]

    # High threat criteria (all must be met):
    if tr_lag1 > 3 and p_lag1 < 0.05 and vif > 2 and acf_lag1 > 0.5:
        return 'high'

    # Moderate threat criteria (any two):
    moderate_flags = [
        tr_lag1 > 2,
        p_lag1 < 0.1,
        vif > 1.5,
        acf_lag1 > 0.3
    ]
    if sum(moderate_flags) >= 2:
        return 'moderate'

    return 'low'


def generate_autocorr_recommendation(threat_level, within_acf, vif, between_ccf):
    """Generate actionable recommendation based on threat level.

    Args:
        threat_level: 'low', 'moderate', or 'high'
        within_acf: (max_lag,) array of within-seed autocorrelations
        vif: Variance inflation factor
        between_ccf: (max_lag,) array of between-seed cross-correlations

    Returns:
        dict with 'use_block_bootstrap' flag and 'message' text
    """
    if threat_level == 'high':
        # Estimate decorrelation time
        block_size = int(np.ceil(1 / max(0.1, 1 - within_acf[0])))
        block_size = min(block_size, 5)

        return {
            'use_block_bootstrap': True,
            'block_size': block_size,
            'message': (
                f"HIGH THREAT (VIF={vif:.2f}): Within-seed autocorrelation "
                f"(ACF[1]={within_acf[0]:.3f}) is significantly higher than "
                f"between-seed correlation (CCF[1]={between_ccf[0]:.3f}). "
                "The current bootstrap method likely UNDERESTIMATES variance.\n\n"
                f"RECOMMENDATION: Use block bootstrap with block_size={block_size} "
                "to preserve temporal structure within seeds. This will produce wider "
                "(more conservative) confidence intervals that properly account for "
                "autocorrelation."
            )
        }

    elif threat_level == 'moderate':
        return {
            'use_block_bootstrap': False,
            'message': (
                f"MODERATE: Some autocorrelation detected (VIF={vif:.2f}). "
                f"Within-seed ACF[1]={within_acf[0]:.3f}, "
                f"between-seed CCF[1]={between_ccf[0]:.3f}. "
                "Bootstrap variance estimates may be slightly optimistic (10-30% too narrow) "
                "but likely acceptable for exploratory analysis.\n\n"
                "SUGGESTION: Interpret confidence intervals cautiously. Consider sensitivity "
                "analysis with block bootstrap if results are borderline significant."
            )
        }

    else:  # low
        return {
            'use_block_bootstrap': False,
            'message': (
                f"LOW THREAT (VIF={vif:.2f}): Autocorrelation structure does not "
                "substantially threaten bootstrap validity. Within-seed and between-seed "
                "correlations are comparable, supporting the exchangeability assumption.\n\n"
                "CONCLUSION: Current bootstrap method is appropriate. Confidence intervals "
                "should be reliable."
            )
        }


def generate_autocorr_interpretation(result):
    """Generate detailed human-readable interpretation.

    Args:
        result: Dict with autocorrelation test results

    Returns:
        Multi-line string with formatted interpretation
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Autocorrelation Diagnostic")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Data: {result['n_runs']} seeds, {result['n_steps']} time points")
    lines.append("")

    lines.append("Within-Seed Autocorrelation (ACF):")
    for lag, acf in enumerate(result['within_seed_acf'], 1):
        lines.append(f"  Lag {lag}: {acf:7.3f}")
    lines.append("")

    lines.append("Between-Seed Cross-Correlation (CCF):")
    for lag, ccf in enumerate(result['between_seed_ccf'], 1):
        lines.append(f"  Lag {lag}: {ccf:7.3f}")
    lines.append("")

    lines.append("Threat Ratio (Within/Between):")
    for lag, (tr, pval) in enumerate(zip(result['threat_ratio'], result['p_values']), 1):
        sig = " ***" if pval < 0.01 else " **" if pval < 0.05 else " *" if pval < 0.1 else ""
        lines.append(f"  Lag {lag}: {tr:7.2f} (p={pval:.3f}){sig}")
    lines.append("")

    lines.append(f"Effective Sample Size (within-seed): {result['n_eff']:.1f} / {result['n_steps']}")
    lines.append(f"Variance Inflation Factor: {result['vif']:.2f}")
    lines.append("")

    # Interpretation note without prescriptive classification
    lines.append("Interpretation:")
    lines.append("  - Threat ratio ≈ 1: Seeds have similar temporal dynamics")
    lines.append("  - Threat ratio >> 1: Within-seed structure much stronger (mixing seeds problematic)")
    lines.append("  - High VIF: Limited effective observations due to autocorrelation")
    lines.append("  - With only 3 seeds, bootstrap uncertainty estimates should be interpreted cautiously")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def test_autocorrelation_structure(steps, gaps_all_runs, max_lag=5,
                                   n_permutations=1000, verbose=True):
    """Test if within-seed autocorrelation threatens bootstrap validity.

    Tests whether the bootstrap assumption of exchangeability across seeds
    is violated by temporal autocorrelation within seeds.

    The current bootstrap method (lines 480-482) creates virtual trajectories
    by randomly selecting which seed to use at each time point. This assumes
    gap values at each time point are exchangeable across seeds. If successive
    values within a seed are highly correlated (strong temporal structure),
    but correlation between seeds is weak, then mixing seeds across time points
    creates unrealistic trajectories and underestimates variance.

    Args:
        steps: (n_steps,) array of probe evaluation steps (for plotting)
        gaps_all_runs: (n_runs, n_steps) array of gap values
        max_lag: Maximum lag to test (default: 5, adequate for 20-40 time points)
        n_permutations: Number of permutation test iterations (default: 1000)
        verbose: Print detailed diagnostic output

    Returns:
        dict with:
            - within_seed_acf: (max_lag,) array of within-seed autocorrelations
            - between_seed_ccf: (max_lag,) array of between-seed cross-correlations
            - threat_ratio: (max_lag,) array of threat ratios (within/between)
            - p_values: (max_lag,) array of permutation test p-values
            - n_eff: Effective sample size adjusted for autocorrelation
            - vif: Variance inflation factor
            - threat_level: 'low', 'moderate', or 'high'
            - recommendation: Dict with use_block_bootstrap flag and message
            - interpretation: Detailed text explanation
            - steps: Copy of input steps (for plotting)
            - gaps_all_runs: Copy of input gaps (for plotting)
            - n_runs, n_steps, max_lag: Metadata
            - success: True if test completed successfully
    """
    n_runs, n_steps = gaps_all_runs.shape

    # Handle edge cases
    if n_steps < 10:
        return {
            'success': False,
            'message': 'Time series too short (n_steps < 10) for reliable autocorrelation estimation'
        }

    # Adjust max_lag if necessary
    max_lag = min(max_lag, n_steps // 3)

    if verbose:
        print(f"  Computing autocorrelation structure ({n_runs} seeds, {n_steps} time points)...")

    # Step 1: Compute ACF and CCF
    within_acf = compute_within_acf(gaps_all_runs, max_lag)
    between_ccf = compute_between_ccf(gaps_all_runs, max_lag)

    # Step 2: Compute threat ratios
    epsilon = 1e-6
    threat_ratio = within_acf / (np.abs(between_ccf) + epsilon)
    threat_ratio = np.minimum(threat_ratio, 100.0)

    if verbose:
        print(f"    Within-seed ACF[1]: {within_acf[0]:.3f}")
        print(f"    Between-seed CCF[1]: {between_ccf[0]:.3f}")
        print(f"    Threat ratio[1]: {threat_ratio[0]:.2f}")

    # Step 3: Permutation test
    if verbose:
        print(f"    Running permutation test ({n_permutations} iterations)...")

    p_values, null_distribution = permutation_test_autocorr(
        gaps_all_runs, max_lag, n_permutations, threat_ratio
    )

    # Step 4: Effective sample size
    n_eff, vif = compute_effective_n(within_acf, n_steps)

    if verbose:
        print(f"    Effective sample size: {n_eff:.1f} (VIF={vif:.2f})")
        print(f"    Permutation p-value[1]: {p_values[0]:.3f}")

    # Step 5: Classify threat level
    threat_level = classify_threat(threat_ratio, p_values, vif, within_acf)

    # Step 6: Generate recommendations
    recommendation = generate_autocorr_recommendation(threat_level, within_acf, vif, between_ccf)

    # Step 7: Generate interpretation text
    result = {
        'success': True,
        'within_seed_acf': within_acf,
        'between_seed_ccf': between_ccf,
        'threat_ratio': threat_ratio,
        'p_values': p_values,
        'null_distribution': null_distribution,
        'n_eff': n_eff,
        'vif': vif,
        'threat_level': threat_level,
        'recommendation': recommendation,
        'steps': steps.copy() if isinstance(steps, np.ndarray) else np.array(steps),
        'gaps_all_runs': gaps_all_runs.copy(),
        'n_runs': n_runs,
        'n_steps': n_steps,
        'max_lag': max_lag
    }

    interpretation = generate_autocorr_interpretation(result)
    result['interpretation'] = interpretation

    if verbose:
        print(f"\n{interpretation}")

    return result


# =============================================================================
# Visualization
# =============================================================================

def plot_analysis(steps, gaps_all_runs, bootstrap_result, title, output_path):
    """Create analysis figure.

    Panel 1: Raw trajectories + mean + best fit
    Panel 2: Bootstrap distribution of gap_inf with CI
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = np.array(steps)
    gaps_all_runs = np.array(gaps_all_runs)
    mean_gap = np.mean(gaps_all_runs, axis=0)

    # Panel 1: Trajectories and fit
    ax = axes[0]

    # Individual runs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, gap_run in enumerate(gaps_all_runs):
        ax.plot(steps, gap_run, 'o-', color=colors[i % len(colors)],
                alpha=0.4, markersize=3, linewidth=1, label=f'Run {i+1}')

    # Mean
    ax.plot(steps, mean_gap, 'ko-', markersize=4, linewidth=2, label='Mean')

    # Best fit curve
    if bootstrap_result['success']:
        fit = bootstrap_result['point_fit']
        t_fine = np.linspace(steps[0], steps[-1] * 1.5, 200)
        ax.plot(t_fine, fit['model_func'](t_fine), 'r-', linewidth=2,
                label=f"{fit['name']} fit")

        # Asymptote
        ax.axhline(fit['gap_inf'], color='red', linestyle='--', alpha=0.7,
                   label=f"gap_inf = {fit['gap_inf']:.4f}")

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(steps[-1], color='orange', linestyle='--', alpha=0.5,
               label='End of training')

    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Gap in Probe R²', fontsize=11)
    ax.set_title(f'{title}\nGap Trajectory', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Bootstrap distribution
    ax = axes[1]

    if bootstrap_result['success']:
        samples = bootstrap_result['bootstrap_samples']

        ax.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue',
                edgecolor='white')

        # CI lines
        ci = bootstrap_result['ci_95']
        ax.axvline(ci[0], color='red', linestyle='--', linewidth=2,
                   label=f'95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]')
        ax.axvline(ci[1], color='red', linestyle='--', linewidth=2)

        # Point estimate
        ax.axvline(bootstrap_result['gap_inf'], color='black', linewidth=2,
                   label=f'Point estimate: {bootstrap_result["gap_inf"]:.4f}')

        # Zero line
        ax.axvline(0, color='gray', linestyle=':', linewidth=2, label='Zero')

        # Annotation
        if ci[0] > 0:
            conclusion = "CI excludes zero\n(persistent advantage)"
            color = 'green'
        elif ci[1] < 0:
            conclusion = "CI excludes zero\n(persistent disadvantage)"
            color = 'red'
        else:
            frac = bootstrap_result['frac_positive']
            conclusion = f"CI includes zero\n({100*frac:.0f}% positive)"
            color = 'orange'

        ax.text(0.95, 0.95, conclusion, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    ax.set_xlabel('Asymptotic Gap (gap_inf)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Bootstrap Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot to {output_path}")


def plot_exp_vs_power_analysis(steps, gaps_all_runs, bootstrap_result, title, output_path):
    """Create analysis figure comparing exponential and power law models.

    Panel 1: Raw trajectories + both model fits
    Panel 2: Bootstrap distributions for both models side-by-side
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = np.array(steps)
    gaps_all_runs = np.array(gaps_all_runs)
    mean_gap = np.mean(gaps_all_runs, axis=0)

    # Panel 1: Trajectories and both fits
    ax = axes[0]

    # Individual runs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, gap_run in enumerate(gaps_all_runs):
        ax.plot(steps, gap_run, 'o-', color=colors[i % len(colors)],
                alpha=0.4, markersize=3, linewidth=1, label=f'Run {i+1}')

    # Mean
    ax.plot(steps, mean_gap, 'ko-', markersize=4, linewidth=2, label='Mean')

    # Both model fits
    t_fine = np.linspace(steps[0], steps[-1] * 1.5, 200)

    # Exponential fit
    exp_fit = fit_single_exp(steps, mean_gap)
    if exp_fit['success']:
        ax.plot(t_fine, exp_fit['model_func'](t_fine), '-', color='#377eb8',
                linewidth=2.5, label=f"Exponential (gap_inf={exp_fit['gap_inf']:.4f})")
        ax.axhline(exp_fit['gap_inf'], color='#377eb8', linestyle='--', alpha=0.5)

    # Power law fit
    power_fit = fit_power_law(steps, mean_gap)
    if power_fit['success']:
        ax.plot(t_fine, power_fit['model_func'](t_fine), '-', color='#4daf4a',
                linewidth=2.5, label=f"Power law (gap_inf={power_fit['gap_inf']:.4f})")
        ax.axhline(power_fit['gap_inf'], color='#4daf4a', linestyle='--', alpha=0.5)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(steps[-1], color='orange', linestyle='--', alpha=0.5,
               label='End of training')

    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Gap in Probe R²', fontsize=11)
    ax.set_title(f'{title}\nModel Comparison', fontsize=12)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Panel 2: Bootstrap distributions
    ax = axes[1]

    model_names = list(bootstrap_result['per_model'].keys())
    model_colors = {'Single exponential': '#377eb8', 'Power law': '#4daf4a'}

    positions = []
    for idx, model_name in enumerate(model_names):
        m = bootstrap_result['per_model'][model_name]
        samples = m['bootstrap_samples']
        position = idx
        positions.append(position)

        # Violin plot
        parts = ax.violinplot([samples], positions=[position], widths=0.7,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(model_colors.get(model_name, 'gray'))
            pc.set_alpha(0.6)

        # Mean and CI
        gap_inf = m['gap_inf']
        ci_low, ci_high = m['ci_95']

        ax.plot(position, gap_inf, 'o', color='black', markersize=8, zorder=10)
        ax.plot([position, position], [ci_low, ci_high], 'k-', linewidth=3, zorder=9)

        # Annotate
        is_best = model_name == bootstrap_result['bic_selected']
        marker = " (BIC)" if is_best else ""
        ax.text(position, ax.get_ylim()[1] * 0.95, f"{model_name.split()[0]}{marker}",
                ha='center', va='top', fontsize=9, fontweight='bold' if is_best else 'normal')

    ax.axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero gap')
    ax.set_xticks(positions)
    ax.set_xticklabels([])
    ax.set_ylabel('Asymptotic Gap (gap_inf)', fontsize=11)
    ax.set_title('Bootstrap Distributions', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Agreement indicator
    agreement_text = "Models AGREE" if bootstrap_result['agreement'] else "Models DISAGREE"
    agreement_color = 'green' if bootstrap_result['agreement'] else 'red'
    ax.text(0.5, 0.05, agreement_text, transform=ax.transAxes,
            fontsize=11, ha='center', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=agreement_color, alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot to {output_path}")


def plot_all_models_analysis(steps, gaps_all_runs, multi_result, title, output_path):
    """Create analysis figure showing ALL model fits and their bootstrap distributions.

    Panel 1: Raw trajectories + all model fits
    Panel 2: Bootstrap distributions for each model (violin or overlaid histograms)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    steps = np.array(steps)
    gaps_all_runs = np.array(gaps_all_runs)
    mean_gap = np.mean(gaps_all_runs, axis=0)

    # Model colors
    model_colors = {
        'Double exponential': '#e41a1c',
        'Single exponential': '#377eb8',
        'Power law': '#4daf4a',
    }

    # Panel 1: Trajectories and ALL fits
    ax = axes[0]

    # Individual runs (light gray)
    for i, gap_run in enumerate(gaps_all_runs):
        ax.plot(steps, gap_run, 'o-', color='gray', alpha=0.3, markersize=2, linewidth=0.5)

    # Mean
    ax.plot(steps, mean_gap, 'ko-', markersize=4, linewidth=2, label='Mean', zorder=10)

    # All model fit curves
    if multi_result['success']:
        t_fine = np.linspace(steps[0], steps[-1] * 1.5, 200)
        bic_selected = multi_result['bic_selected']

        for model_name, fit in multi_result['point_fits'].items():
            if not fit['success']:
                continue
            color = model_colors.get(model_name, 'gray')
            is_best = model_name == bic_selected
            lw = 2.5 if is_best else 1.5
            alpha = 1.0 if is_best else 0.6
            label = f"{model_name}" + (" (BIC best)" if is_best else "")

            ax.plot(t_fine, fit['model_func'](t_fine), '-', color=color,
                    linewidth=lw, alpha=alpha, label=label)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(steps[-1], color='orange', linestyle='--', alpha=0.5, label='End of training')

    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Gap in Probe R²', fontsize=11)
    ax.set_title(f'{title}\nAll Model Fits', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Bootstrap distributions for each model
    ax = axes[1]

    if multi_result['success']:
        per_model = multi_result['per_model']

        # Violin-style: plot each model's distribution
        positions = list(range(len(per_model)))
        model_names = list(per_model.keys())

        violin_data = []
        for model_name in model_names:
            violin_data.append(per_model[model_name]['bootstrap_samples'])

        parts = ax.violinplot(violin_data, positions=positions, showmeans=True,
                              showextrema=True, showmedians=False)

        # Color the violins
        for i, (pc, model_name) in enumerate(zip(parts['bodies'], model_names)):
            color = model_colors.get(model_name, 'gray')
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Add point estimates and CIs
        for i, model_name in enumerate(model_names):
            m = per_model[model_name]
            ci = m['ci_95']
            # CI whiskers
            ax.vlines(i, ci[0], ci[1], color='black', linewidth=2)
            ax.hlines([ci[0], ci[1]], i-0.1, i+0.1, color='black', linewidth=2)
            # Point estimate
            ax.scatter([i], [m['gap_inf']], color='black', s=50, zorder=10)

        ax.set_xticks(positions)
        ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=9)
        ax.axhline(0, color='gray', linestyle=':', linewidth=2, label='Zero')

        # Add agreement annotation
        agreement = multi_result['agreement']
        if agreement:
            combined = multi_result['combined_ci']
            text = f"Models AGREE\nCombined CI: [{combined[0]:.4f}, {combined[1]:.4f}]"
            color = 'green' if combined[0] > 0 else ('red' if combined[1] < 0 else 'orange')
        else:
            text = "Models DISAGREE\n(CIs don't overlap)"
            color = 'red'

        ax.text(0.95, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    ax.set_ylabel('Asymptotic Gap (gap_inf)', fontsize=11)
    ax.set_title('Bootstrap Distributions by Model', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot to {output_path}")


def plot_autocorrelation_diagnostic(result, output_path):
    """Create diagnostic visualization for autocorrelation test.

    Creates a 3-panel figure:
    1. ACF vs CCF comparison (line plot)
    2. Threat ratio by lag (bar plot with significance colors)
    3. Raw time series by seed (shows whether seeds move together)

    Args:
        result: Dict returned by test_autocorrelation_structure()
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: ACF vs CCF
    ax = axes[0]
    lags = np.arange(1, len(result['within_seed_acf']) + 1)

    ax.plot(lags, result['within_seed_acf'], 'o-', color='steelblue',
            linewidth=2.5, markersize=8, label='Within-seed ACF', zorder=3)
    ax.plot(lags, result['between_seed_ccf'], 's--', color='darkorange',
            linewidth=2.5, markersize=7, label='Between-seed CCF', zorder=3)

    # Fill between to show gap
    fill_color = 'red' if result['threat_level'] == 'high' else 'orange' if result['threat_level'] == 'moderate' else 'gray'
    ax.fill_between(lags, result['within_seed_acf'], result['between_seed_ccf'],
                    alpha=0.2, color=fill_color)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Autocorrelation Structure', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(lags)

    # Panel 2: Threat Ratio with significance colors
    ax = axes[1]

    # Color by significance
    colors = []
    for p in result['p_values']:
        if p < 0.01:
            colors.append('#d62728')  # red
        elif p < 0.05:
            colors.append('#ff7f0e')  # orange
        elif p < 0.1:
            colors.append('#ffbb78')  # light orange
        else:
            colors.append('#2ca02c')  # green

    bars = ax.bar(lags, result['threat_ratio'], color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)

    # Threshold lines
    ax.axhline(1, color='gray', linestyle=':', linewidth=2, label='No threat (ratio=1)', zorder=1)
    ax.axhline(2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Moderate threshold', zorder=1)
    ax.axhline(3, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='High threshold', zorder=1)

    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Threat Ratio (Within/Between)', fontsize=12)
    ax.set_title('Bootstrap Threat Assessment', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(lags)
    ax.set_ylim(bottom=0)

    # Panel 3: Raw trajectories
    ax = axes[2]
    steps = result['steps']
    gaps = result['gaps_all_runs']
    n_runs = gaps.shape[0]

    colors_seeds = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(n_runs):
        ax.plot(steps, gaps[i, :], 'o-', color=colors_seeds[i % len(colors_seeds)],
                alpha=0.7, linewidth=2, markersize=5, label=f'Seed {i+1}')

    # Add mean trajectory
    mean_gaps = np.mean(gaps, axis=0)
    ax.plot(steps, mean_gaps, 'k--', linewidth=2.5, alpha=0.7, label='Mean', zorder=10)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Gap Value (R²)', fontsize=12)
    ax.set_title('Raw Time Series by Seed', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Overall title
    vif_text = f"VIF={result['vif']:.2f}"
    tr_text = f"TR={result['threat_ratio'][0]:.2f}"
    acf_text = f"ACF={result['within_seed_acf'][0]:.2f}"
    ccf_text = f"CCF={result['between_seed_ccf'][0]:.2f}"

    fig.suptitle(
        f"Autocorrelation Diagnostic: {acf_text}, {ccf_text}, {tr_text}, {vif_text}",
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved autocorrelation diagnostic plot to {output_path}")


# =============================================================================
# Horizon Analysis
# =============================================================================

def analyze_horizon_stability(histories, cond_a, cond_b, label, layer='2',
                               horizons=None, n_bootstrap=300, verbose=True):
    """Analyze how gap_inf estimates change with different data horizons.

    Compares all 3 models (single exp, double exp, power law) to see which
    gives stable predictions with less training data.

    Args:
        histories: Dict of training histories
        cond_a, cond_b: Condition patterns to compare
        label: Display label for this comparison
        layer: Which layer to analyze
        horizons: List of max training steps to use (default: fractions [0.2, 0.4, 0.6, 0.8, 1.0] of full data)
        n_bootstrap: Bootstrap iterations per horizon per model

    Returns:
        Dict with results per horizon per model
    """
    # Extract full data
    steps, gaps, seeds = extract_gap_data(histories, cond_a, cond_b, layer)
    steps = np.array(steps)
    gaps = np.array(gaps)

    if horizons is None:
        # Use fractions of full data to align with cross-validation
        max_steps = steps[-1]
        horizons = [int(max_steps * frac) for frac in [0.2, 0.4, 0.6, 0.8, 1.0]]

    # Ensure horizons don't exceed data
    horizons = [h for h in horizons if h <= steps[-1]]
    if steps[-1] not in horizons:
        horizons.append(steps[-1])
    horizons = sorted(horizons)

    if verbose:
        print(f"\n  Analyzing {len(horizons)} horizons: {horizons}")

    fit_funcs = {
        'Single exponential': fit_single_exp,
        'Double exponential': fit_double_exp,
        'Power law': fit_power_law,
    }

    results = {model: [] for model in fit_funcs}
    results['horizons'] = horizons

    for horizon in horizons:
        # Truncate data to this horizon
        mask = steps <= horizon
        steps_trunc = steps[mask]
        gaps_trunc = gaps[:, mask]

        if verbose:
            print(f"\n  Horizon {horizon} ({len(steps_trunc)} points):")

        mean_gap = np.mean(gaps_trunc, axis=0)
        n_runs, n_steps = gaps_trunc.shape

        for model_name, fit_func in fit_funcs.items():
            # Point estimate
            point_fit = fit_func(steps_trunc, mean_gap)

            if not point_fit['success']:
                results[model_name].append({
                    'horizon': horizon,
                    'gap_inf': np.nan,
                    'ci_95': (np.nan, np.nan),
                    'success': False
                })
                if verbose:
                    print(f"    {model_name}: FAILED")
                continue

            # Bootstrap
            bootstrap_samples = []
            for _ in range(n_bootstrap):
                run_choices = np.random.randint(0, n_runs, size=n_steps)
                virtual_gap = gaps_trunc[run_choices, np.arange(n_steps)]
                result = fit_func(steps_trunc, virtual_gap)
                if result['success']:
                    bootstrap_samples.append(result['gap_inf'])
                else:
                    bootstrap_samples.append(np.mean(virtual_gap[-3:]))

            bootstrap_samples = np.array(bootstrap_samples)
            ci_95 = (np.percentile(bootstrap_samples, 2.5), np.percentile(bootstrap_samples, 97.5))

            results[model_name].append({
                'horizon': horizon,
                'gap_inf': point_fit['gap_inf'],
                'ci_95': ci_95,
                'se': np.std(bootstrap_samples),
                'success': True
            })

            if verbose:
                print(f"    {model_name}: gap_inf={point_fit['gap_inf']:.4f} CI=[{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

    return results


def plot_horizon_stability(results, output_path, title="Horizon Stability"):
    """Plot gap_inf estimates vs training horizon for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_colors = {
        'Single exponential': '#377eb8',
        'Double exponential': '#e41a1c',
        'Power law': '#4daf4a',
    }

    horizons = results['horizons']
    x = np.arange(len(horizons))

    for model_name in ['Single exponential', 'Double exponential', 'Power law']:
        if model_name not in results:
            continue

        model_results = results[model_name]
        gap_infs = [r['gap_inf'] for r in model_results]
        ci_lows = [r['ci_95'][0] for r in model_results]
        ci_highs = [r['ci_95'][1] for r in model_results]

        color = model_colors[model_name]
        ax.plot(x, gap_infs, 'o-', color=color, linewidth=2, markersize=8, label=model_name)
        ax.fill_between(x, ci_lows, ci_highs, color=color, alpha=0.15)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h//1000}k' for h in horizons])
    ax.set_xlabel('Training Horizon (steps)', fontsize=11)
    ax.set_ylabel('Estimated Asymptotic Gap (gap_inf)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved horizon stability plot to {output_path}")


def cross_validate_models(steps, gaps, train_fractions=None, verbose=True):
    """Cross-validate models using out-of-sample prediction on held-out data.

    For each train fraction:
    - Fit each model to first train_fraction of mean trajectory
    - Extrapolate to remaining data
    - Compute prediction error on held-out data

    This tests which model better predicts unseen future behavior.

    Args:
        steps: (n_steps,) array of training steps
        gaps: (n_runs, n_steps) array of gap values
        train_fractions: List of fractions for train/test splits (default: [0.2, 0.4, 0.6, 0.8])
        verbose: Print progress

    Returns:
        dict with:
            - train_fractions: array of train fractions tested
            - models: dict mapping model_name -> {
                'test_rmse': array of RMSE on test data for each fraction
                'test_mae': array of MAE on test data for each fraction
                'mean_test_rmse': average RMSE across all splits
                'mean_test_mae': average MAE across all splits
              }
            - best_model: name of model with lowest mean test RMSE
    """
    steps = np.array(steps, dtype=float)
    gaps = np.array(gaps)
    n_runs, n_steps = gaps.shape
    mean_gap = np.mean(gaps, axis=0)

    if train_fractions is None:
        train_fractions = [0.2, 0.4, 0.6, 0.8]

    fit_funcs = {
        'Single exponential': fit_single_exp,
        'Double exponential': fit_double_exp,
        'Power law': fit_power_law,
        'Double power law': fit_double_power_law,
    }

    results = {
        'train_fractions': train_fractions,
        'models': {name: {'test_rmse': [], 'test_mae': []} for name in fit_funcs}
    }

    if verbose:
        print(f"\n  Cross-validating models with {len(train_fractions)} train/test splits...")

    for frac in train_fractions:
        n_train = int(n_steps * frac)
        n_test = n_steps - n_train

        if n_test < 3:
            if verbose:
                print(f"    Skipping fraction {frac:.1f} (too few test points)")
            continue

        # Split data
        steps_train = steps[:n_train]
        steps_test = steps[n_train:]
        gap_train = mean_gap[:n_train]
        gap_test = mean_gap[n_train:]

        if verbose:
            print(f"\n    Train fraction {frac:.1f}: {n_train} train, {n_test} test points")
            print(f"      Train range: [{steps_train[0]:.0f}, {steps_train[-1]:.0f}]")
            print(f"      Test range: [{steps_test[0]:.0f}, {steps_test[-1]:.0f}]")

        for model_name, fit_func in fit_funcs.items():
            # Fit to training data
            fit_result = fit_func(steps_train, gap_train)

            if not fit_result['success']:
                if verbose:
                    print(f"        {model_name}: FAILED")
                results['models'][model_name]['test_rmse'].append(np.nan)
                results['models'][model_name]['test_mae'].append(np.nan)
                continue

            # Extrapolate to test data
            pred_test = fit_result['model_func'](steps_test)

            # Compute errors
            rmse = np.sqrt(np.mean((gap_test - pred_test)**2))
            mae = np.mean(np.abs(gap_test - pred_test))

            results['models'][model_name]['test_rmse'].append(rmse)
            results['models'][model_name]['test_mae'].append(mae)

            if verbose:
                print(f"        {model_name}: RMSE={rmse:.5f}, MAE={mae:.5f}")

    # Compute means and identify best model
    best_rmse = np.inf
    best_model = None

    for model_name, model_results in results['models'].items():
        test_rmse = np.array([r for r in model_results['test_rmse'] if not np.isnan(r)])
        test_mae = np.array([r for r in model_results['test_mae'] if not np.isnan(r)])

        if len(test_rmse) > 0:
            mean_rmse = np.mean(test_rmse)
            mean_mae = np.mean(test_mae)
            model_results['mean_test_rmse'] = mean_rmse
            model_results['mean_test_mae'] = mean_mae

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = model_name

    results['best_model'] = best_model

    if verbose:
        print(f"\n  Summary of out-of-sample prediction errors:")
        for model_name, model_results in results['models'].items():
            if 'mean_test_rmse' in model_results:
                marker = " <-- BEST" if model_name == best_model else ""
                print(f"    {model_name}: Mean RMSE={model_results['mean_test_rmse']:.5f}, "
                      f"Mean MAE={model_results['mean_test_mae']:.5f}{marker}")

    return results


def plot_cross_validation(cv_results, output_path, title="Model Cross-Validation"):
    """Plot cross-validation results showing prediction errors vs train fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_colors = {
        'Single exponential': '#377eb8',
        'Double exponential': '#e41a1c',
        'Power law': '#4daf4a',
        'Double power law': '#984ea3',
    }

    train_fracs = cv_results['train_fractions']

    # Panel 1: RMSE
    ax = axes[0]
    for model_name, model_results in cv_results['models'].items():
        rmse_vals = model_results['test_rmse']
        color = model_colors.get(model_name, 'gray')

        # Only plot where we have valid data
        valid_mask = [not np.isnan(r) for r in rmse_vals]
        valid_fracs = [f for f, v in zip(train_fracs, valid_mask) if v]
        valid_rmse = [r for r, v in zip(rmse_vals, valid_mask) if v]

        if valid_rmse:
            marker = 'o' if model_name == cv_results['best_model'] else 's'
            linewidth = 2.5 if model_name == cv_results['best_model'] else 1.5
            ax.plot(valid_fracs, valid_rmse, marker=marker, color=color,
                   linewidth=linewidth, markersize=8, label=model_name)

    ax.set_xlabel('Train Fraction', fontsize=11)
    ax.set_ylabel('Test RMSE (out-of-sample)', fontsize=11)
    ax.set_title('Prediction Error vs Training Data', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: MAE
    ax = axes[1]
    for model_name, model_results in cv_results['models'].items():
        mae_vals = model_results['test_mae']
        color = model_colors.get(model_name, 'gray')

        valid_mask = [not np.isnan(m) for m in mae_vals]
        valid_fracs = [f for f, v in zip(train_fracs, valid_mask) if v]
        valid_mae = [m for m, v in zip(mae_vals, valid_mask) if v]

        if valid_mae:
            marker = 'o' if model_name == cv_results['best_model'] else 's'
            linewidth = 2.5 if model_name == cv_results['best_model'] else 1.5
            ax.plot(valid_fracs, valid_mae, marker=marker, color=color,
                   linewidth=linewidth, markersize=8, label=model_name)

    ax.set_xlabel('Train Fraction', fontsize=11)
    ax.set_ylabel('Test MAE (out-of-sample)', fontsize=11)
    ax.set_title('Mean Absolute Error vs Training Data', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved cross-validation plot to {output_path}")


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_comparison(histories, cond_a, cond_b, label, layer='2',
                       n_bootstrap=5000, plots_dir='plots', all_models=False):
    """Full analysis for one comparison pair.

    Args:
        histories: Dict of training histories
        cond_a, cond_b: Condition patterns to compare
        label: Display label for this comparison
        layer: Which layer to analyze (default: '2' = embedded layer)
        n_bootstrap: Number of bootstrap iterations (default: 5000)
        plots_dir: Directory for output plots
        all_models: If True, bootstrap ALL 4 models; else bootstrap exp+power (default)
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Extract data
    try:
        steps, gaps, seeds = extract_gap_data(histories, cond_a, cond_b, layer)
    except ValueError as e:
        print(f"  Error: {e}")
        return None

    print(f"  Seeds: {seeds}")
    print(f"  Probe evaluations: {len(steps)} (steps {steps[0]} to {steps[-1]})")
    print(f"  Layer: {layer}")

    # Quick stats
    mean_gap = np.mean(gaps, axis=0)
    print(f"\n  Current gap (last eval): {mean_gap[-1]:.4f}")
    print(f"  Initial gap (first eval): {mean_gap[0]:.4f}")

    # Check monotonicity
    n_decreasing = np.sum(np.diff(mean_gap) < 0)
    print(f"  Gap decreasing in {n_decreasing}/{len(mean_gap)-1} intervals")

    # Bootstrap - either single best model or all models
    if all_models:
        result = bootstrap_all_models(steps, gaps, n_bootstrap, verbose=True)

        if not result['success']:
            print("  Bootstrap failed")
            return None

        # Report per-model results
        print(f"\n  === Results (All Models) ===")
        for model_name, m in result['per_model'].items():
            is_best = model_name == result['bic_selected']
            best_marker = " [BIC best]" if is_best else ""
            print(f"  {model_name}{best_marker}:")
            print(f"    gap_inf: {m['gap_inf']:.4f} (SE: {m['se']:.4f})")
            print(f"    95% CI: [{m['ci_95'][0]:.4f}, {m['ci_95'][1]:.4f}]")

        # Report agreement
        print(f"\n  Model Agreement: {'YES' if result['agreement'] else 'NO'}")
        if result['agreement']:
            combined = result['combined_ci']
            print(f"  Combined CI: [{combined[0]:.4f}, {combined[1]:.4f}]")
            if combined[0] > 0:
                print(f"  --> ALL MODELS AGREE: Persistent advantage")
            elif combined[1] < 0:
                print(f"  --> ALL MODELS AGREE: Persistent disadvantage")
            else:
                print(f"  --> ALL MODELS AGREE: No significant persistent gap")
        else:
            print(f"  --> Models disagree - interpret with caution")

        # Plot with all models
        output_path = Path(plots_dir) / f"asymptotic_gap_all_models_{cond_a.replace('+', '_').replace('(', '').replace(')', '')}_vs_{cond_b.replace('+', '_').replace('(', '').replace(')', '')}.png"
        plot_all_models_analysis(steps, gaps, result, label, output_path)

        # For summary table, use BIC-selected model
        bic_model = result['per_model'][result['bic_selected']]
        result['gap_inf'] = bic_model['gap_inf']
        result['ci_95'] = bic_model['ci_95']
        result['frac_positive'] = bic_model['frac_positive']

    else:
        # Bootstrap exponential and power law models
        result = bootstrap_exp_and_power(steps, gaps, n_bootstrap, verbose=True)

        if not result['success']:
            print("  Bootstrap failed")
            return None

        # Report per-model results
        print(f"\n  === Results (Exponential vs Power Law) ===")
        for model_name, m in result['per_model'].items():
            is_best = model_name == result['bic_selected']
            best_marker = " [BIC best]" if is_best else ""
            print(f"  {model_name}{best_marker}:")
            print(f"    gap_inf: {m['gap_inf']:.4f} (SE: {m['se']:.4f})")
            print(f"    95% CI: [{m['ci_95'][0]:.4f}, {m['ci_95'][1]:.4f}]")
            if m['n_failures'] > 0:
                print(f"    Note: {m['n_failures']}/{n_bootstrap} fits failed (used fallback)")

        # Report agreement
        print(f"\n  Model Agreement: {'YES' if result['agreement'] else 'NO'}")
        if result['agreement']:
            print(f"  --> Both models agree on sign/significance of persistent gap")
        else:
            print(f"  --> Models disagree - suggests model uncertainty or regime transition")

        # Plot comparison
        comparison_path = Path(plots_dir) / f"asymptotic_gap_comparison_{cond_a.replace('+', '_').replace('(', '').replace(')', '')}_vs_{cond_b.replace('+', '_').replace('(', '').replace(')', '')}.png"
        plot_exp_vs_power_analysis(steps, gaps, result, label, comparison_path)

        # Plot individual models
        for model_name, model_data in result['per_model'].items():
            model_slug = model_name.lower().replace(' ', '_')
            individual_path = Path(plots_dir) / f"asymptotic_gap_{model_slug}_{cond_a.replace('+', '_').replace('(', '').replace(')', '')}_vs_{cond_b.replace('+', '_').replace('(', '').replace(')', '')}.png"

            # Create a result dict compatible with plot_analysis
            individual_result = {
                'success': True,
                'gap_inf': model_data['gap_inf'],
                'ci_95': model_data['ci_95'],
                'se': model_data['se'],
                'bootstrap_samples': model_data['bootstrap_samples'],
                'frac_positive': model_data['frac_positive'],
                'n_fit_failures': model_data['n_failures'],
                'point_fit': fit_single_exp(steps, np.mean(gaps, axis=0)) if 'exponential' in model_name.lower() else fit_power_law(steps, np.mean(gaps, axis=0))
            }
            plot_analysis(steps, gaps, individual_result, f"{label}\n({model_name})", individual_path)

        # For summary table, use BIC-selected model
        bic_model = result['per_model'][result['bic_selected']]
        result['gap_inf'] = bic_model['gap_inf']
        result['ci_95'] = bic_model['ci_95']
        result['frac_positive'] = bic_model['frac_positive']

    # Autocorrelation diagnostic
    print("\n  === Autocorrelation Diagnostic ===")
    autocorr_result = test_autocorrelation_structure(
        steps, gaps, max_lag=5, n_permutations=1000, verbose=True
    )

    if autocorr_result['success']:
        # Save diagnostic plot
        safe_label = label.replace(' ', '_').replace('/', '_').replace('+', '_').replace('(', '').replace(')', '')
        autocorr_plot_path = Path(plots_dir) / f"autocorr_diagnostic_{safe_label}.png"
        plot_autocorrelation_diagnostic(autocorr_result, autocorr_plot_path)

        # Store summary for reporting
        result['autocorr_diagnostic'] = {
            'threat_level': autocorr_result['threat_level'],
            'vif': float(autocorr_result['vif']),
            'within_acf_lag1': float(autocorr_result['within_seed_acf'][0]),
            'between_ccf_lag1': float(autocorr_result['between_seed_ccf'][0]),
            'threat_ratio_lag1': float(autocorr_result['threat_ratio'][0]),
            'p_value_lag1': float(autocorr_result['p_values'][0]),
            'recommendation': autocorr_result['recommendation']['message']
        }

    # Model cross-validation
    print("\n  === Model Cross-Validation ===")
    cv_result = cross_validate_models(steps, gaps, verbose=True)

    # Save cross-validation plot
    cv_plot_path = Path(plots_dir) / f"model_cv_{safe_label}.png"
    plot_cross_validation(cv_result, cv_plot_path, title=f"Model Cross-Validation: {label}")

    # Store best model info
    result['cross_validation'] = {
        'best_model': cv_result['best_model'],
        'mean_test_rmse': {name: float(model_data['mean_test_rmse'])
                          for name, model_data in cv_result['models'].items()
                          if 'mean_test_rmse' in model_data}
    }

    return result


def print_summary(results):
    """Print summary table of all comparisons."""
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    print(f"\n{'Comparison':<30} {'gap_inf':>10} {'95% CI':>20} {'Significant?':>15}")
    print("-"*80)

    for label, result in results.items():
        if result is None:
            print(f"{label:<30} {'FAILED':>10}")
            continue

        gap_inf = result['gap_inf']
        ci = result['ci_95']
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"

        if ci[0] > 0:
            sig = "Yes (positive)"
        elif ci[1] < 0:
            sig = "Yes (negative)"
        else:
            sig = "No"

        print(f"{label:<30} {gap_inf:>10.4f} {ci_str:>20} {sig:>15}")

    print("="*80)

    # Autocorrelation diagnostic summary
    has_autocorr_data = any(result and 'autocorr_diagnostic' in result
                            for result in results.values())

    if has_autocorr_data:
        print("\n" + "="*80)
        print("  AUTOCORRELATION DIAGNOSTIC SUMMARY")
        print("="*80)
        print(f"\n{'Comparison':<30} {'ACF[1]':>8} {'CCF[1]':>8} {'TR[1]':>8} {'VIF':>8} {'n_eff':>8}")
        print("-"*80)

        for label, result in results.items():
            if result and 'autocorr_diagnostic' in result:
                diag = result['autocorr_diagnostic']
                n_eff = result['n_steps'] / diag['vif'] if 'n_steps' in result else 61 / diag['vif']
                print(f"{label:<30} {diag['within_acf_lag1']:>8.3f} {diag['between_ccf_lag1']:>8.3f} "
                      f"{diag['threat_ratio_lag1']:>8.2f} {diag['vif']:>8.2f} {n_eff:>8.1f}")

        print("="*80)
        print("\nColumn definitions:")
        print("  ACF[1]: Within-seed autocorrelation at lag 1")
        print("  CCF[1]: Between-seed cross-correlation at lag 1")
        print("  TR[1]: Threat ratio (ACF/CCF) at lag 1")
        print("  VIF: Variance inflation factor (time points / effective sample size)")
        print("  n_eff: Effective number of independent observations within each seed")

    # Cross-validation summary
    has_cv_data = any(result and 'cross_validation' in result
                      for result in results.values())

    if has_cv_data:
        print("\n" + "="*80)
        print("  MODEL CROSS-VALIDATION SUMMARY")
        print("="*80)
        print(f"\n{'Comparison':<30} {'Best Model':<20} {'RMSE':>12}")
        print("-"*80)

        for label, result in results.items():
            if result and 'cross_validation' in result:
                cv = result['cross_validation']
                best = cv['best_model']
                rmse = cv['mean_test_rmse'].get(best, float('nan'))
                print(f"{label:<30} {best:<20} {rmse:>12.5f}")

        print("="*80)
        print("\nInterpretation:")
        print("  Best Model: Model with lowest mean out-of-sample prediction error")
        print("  RMSE: Root mean squared error on held-out test data")
        print("  Lower RMSE indicates better extrapolation to unseen future training steps")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive post-experiment analysis for embedding study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis (default)
  python analyze_experiment.py

  # Run only specific analyses
  python analyze_experiment.py --asymptotic
  python analyze_experiment.py --layer-comparison
  python analyze_experiment.py --horizon
        """
    )

    # Analysis selection
    analysis = parser.add_argument_group('Analysis Selection')
    analysis.add_argument('--asymptotic', action='store_true',
                         help='Run only asymptotic gap analysis')
    analysis.add_argument('--alpha-sweep', action='store_true',
                         help='Run only alpha sweep analysis')
    analysis.add_argument('--permutation', action='store_true',
                         help='Run only permutation test')
    analysis.add_argument('--layer-comparison', action='store_true',
                         help='Run only layer comparison analysis')
    analysis.add_argument('--horizon', action='store_true',
                         help='Run only horizon stability analysis')

    # Asymptotic options
    asymp = parser.add_argument_group('Asymptotic Analysis Options')
    asymp.add_argument('--all-models', action='store_true',
                      help='Bootstrap all models instead of just BIC-selected')
    asymp.add_argument('--bootstrap', type=int, default=5000,
                      help='Number of bootstrap iterations (default: 5000)')
    asymp.add_argument('--layer', type=str, default='2',
                      help='Layer for asymptotic analysis (default: 2 = embedded layer)')

    # Alpha sweep options
    alpha = parser.add_argument_group('Alpha Sweep Options')
    alpha.add_argument('--n-alphas', type=int, default=100,
                      help='Number of alpha values to test (default: 100)')
    alpha.add_argument('--n-samples', type=int, default=2000,
                      help='Number of samples for probing (default: 2000)')

    # Permutation options
    perm = parser.add_argument_group('Permutation Test Options')
    perm.add_argument('--n-permutations', type=int, default=1000,
                     help='Number of permutations (default: 1000)')
    perm.add_argument('--perm-layer', type=int, default=2,
                     help='Layer for permutation test (default: 2)')

    # I/O options
    io = parser.add_argument_group('Input/Output')
    io.add_argument('--histories', type=str, default='results/histories.json',
                   help='Path to histories JSON file')
    io.add_argument('--checkpoints-dir', type=str, default='checkpoints',
                   help='Directory containing model checkpoints')
    io.add_argument('--data', type=str, default='data/combined_for_experiment.npz',
                   help='Path to game data file')
    io.add_argument('--output-dir', type=str, default='analysis_results',
                   help='Base directory for analysis outputs')

    args = parser.parse_args()

    # Determine which analyses to run
    specific_analysis = any([args.asymptotic, args.alpha_sweep, args.permutation,
                             args.layer_comparison, args.horizon])

    if not specific_analysis:
        # Default: run everything
        args.asymptotic = True
        args.alpha_sweep = True
        args.permutation = True
        args.layer_comparison = True
        args.horizon = True

    # all_models is already set by argparse (False by default)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track what we ran
    analyses_run = []

    # =========================================================================
    # Asymptotic Gap Analysis
    # =========================================================================
    if args.asymptotic:
        print("\n" + "="*80)
        print("  ASYMPTOTIC GAP ANALYSIS")
        print("="*80)

        try:
            histories = load_histories(args.histories)
        except FileNotFoundError:
            print(f"Error: {args.histories} not found. Run the experiment first.")
            return

        print(f"Found {len(histories)} model runs")
        print(f"Bootstrap iterations: {args.bootstrap}")
        if args.all_models:
            print("Mode: Bootstrap all 4 models and check agreement")
        else:
            print("Mode: Bootstrap exponential + power law models")

        asymp_dir = output_dir / 'asymptotic'
        asymp_dir.mkdir(parents=True, exist_ok=True)

        comparisons = [
            ('Large+embed(aux)', 'Large-baseline', 'Embed+Aux vs Baseline'),
            ('Large+embed(aux)', 'Large+embed(noaux)', 'Embed+Aux vs Embed-NoAux'),
            ('Large+embed(noaux)', 'Large-baseline', 'Embed-NoAux vs Baseline'),
        ]

        results = {}
        for cond_a, cond_b, label in comparisons:
            results[label] = analyze_comparison(
                histories, cond_a, cond_b, label,
                layer=args.layer,
                n_bootstrap=args.bootstrap,
                plots_dir=str(asymp_dir),
                all_models=args.all_models
            )

        print_summary(results)
        analyses_run.append('asymptotic')

    # =========================================================================
    # Alpha Sweep Analysis
    # =========================================================================
    if args.alpha_sweep:
        print("\n" + "="*80)
        print("  ALPHA SWEEP ANALYSIS")
        print("="*80)

        checkpoints_dir = Path(args.checkpoints_dir)
        if not checkpoints_dir.exists():
            print(f"Error: Checkpoints directory {checkpoints_dir} not found.")
        else:
            alphas = np.logspace(-6, 6, args.n_alphas)
            alpha_dir = output_dir / 'alpha_sweep'
            run_alpha_sweep_all_models(
                checkpoints_dir=str(checkpoints_dir),
                data_path=args.data,
                output_dir=str(alpha_dir),
                alphas=alphas,
                n_samples=args.n_samples
            )
            analyses_run.append('alpha_sweep')

    # =========================================================================
    # Permutation Test
    # =========================================================================
    if args.permutation:
        print("\n" + "="*80)
        print("  PERMUTATION TEST")
        print("="*80)

        checkpoints_dir = Path(args.checkpoints_dir)
        if not checkpoints_dir.exists():
            print(f"Error: Checkpoints directory {checkpoints_dir} not found.")
        else:
            perm_dir = output_dir / 'permutation'
            perm_dir.mkdir(parents=True, exist_ok=True)

            # Run on the embed+aux model (the one expected to show strongest effect)
            embed_aux_ckpts = list(checkpoints_dir.glob('large_embed_aux_*_best.pt'))
            if embed_aux_ckpts:
                ckpt_path = embed_aux_ckpts[0]  # Use first seed
                print(f"Running permutation test on {ckpt_path.name}...")

                result = permutation_test_sectors(
                    str(ckpt_path),
                    data_path=args.data,
                    n_permutations=args.n_permutations,
                    layer=args.perm_layer,
                    n_samples=args.n_samples
                )

                plot_permutation_test(result, perm_dir / 'permutation_test.png')

                # Save results
                result_json = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                              for k, v in result.items()}
                with open(perm_dir / 'permutation_results.json', 'w') as f:
                    json.dump(result_json, f, indent=2)

                analyses_run.append('permutation')
            else:
                print(f"No large_embed_aux checkpoints found in {checkpoints_dir}")

    # =========================================================================
    # Layer Comparison Analysis
    # =========================================================================
    if args.layer_comparison:
        print("\n" + "="*80)
        print("  LAYER COMPARISON ANALYSIS")
        print("="*80)

        try:
            histories = load_histories(args.histories)
        except FileNotFoundError:
            print(f"Error: {args.histories} not found. Run the experiment first.")
            histories = None

        if histories:
            layer_dir = output_dir / 'layer_comparison'
            layer_dir.mkdir(parents=True, exist_ok=True)

            results = analyze_layer_comparison(histories)

            # Print results
            print("\nLayer 2 Prominence Scores (higher = more concentration):")
            for model_type, score in results['layer_2_prominence'].items():
                print(f"  {model_type}: {score:.3f}")

            print("\nPeak Layers:")
            for model_type, layer in results['peak_layer'].items():
                print(f"  {model_type}: Layer {layer}")

            plot_layer_comparison(results, layer_dir / 'layer_comparison.png')

            # Save results
            results_json = {
                'layer_profiles': {k: {str(kk): vv for kk, vv in v.items()}
                                   for k, v in results['layer_profiles'].items()},
                'peak_layer': results['peak_layer'],
                'layer_2_prominence': results['layer_2_prominence'],
            }
            with open(layer_dir / 'layer_comparison_results.json', 'w') as f:
                json.dump(results_json, f, indent=2)

            analyses_run.append('layer_comparison')

    # =========================================================================
    # Horizon Stability Analysis
    # =========================================================================
    if args.horizon:
        print("\n" + "="*80)
        print("  HORIZON STABILITY ANALYSIS")
        print("="*80)

        try:
            histories = load_histories(args.histories)
        except FileNotFoundError:
            print(f"Error: {args.histories} not found. Run the experiment first.")
            histories = None

        if histories:
            horizon_dir = output_dir / 'horizon'
            horizon_dir.mkdir(parents=True, exist_ok=True)

            # Analyze main comparison (embed+aux vs baseline)
            print("\nEmbed+Aux vs Baseline:")
            results = analyze_horizon_stability(
                histories, 'Large+embed(aux)', 'Large-baseline',
                'Embed+Aux vs Baseline', layer=args.layer
            )
            plot_horizon_stability(results, horizon_dir / 'horizon_embed_aux_vs_baseline.png',
                                   title='Horizon Stability: Embed+Aux vs Baseline')

            # Save results (convert numpy types to native Python for JSON)
            results_json = {
                'horizons': [int(h) for h in results['horizons']],
                'models': {}
            }
            for name in ['Single exponential', 'Double exponential', 'Power law']:
                if name in results:
                    results_json['models'][name] = [
                        {
                            'horizon': int(r['horizon']),
                            'gap_inf': float(r['gap_inf']) if not np.isnan(r['gap_inf']) else None,
                            'ci_95': [float(r['ci_95'][0]), float(r['ci_95'][1])] if r['success'] else None,
                            'se': float(r.get('se', 0)) if r['success'] else None,
                            'success': r['success']
                        }
                        for r in results[name]
                    ]
            with open(horizon_dir / 'horizon_results.json', 'w') as f:
                json.dump(results_json, f, indent=2)

            analyses_run.append('horizon')

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print("="*80)
    print(f"Analyses run: {', '.join(analyses_run)}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
