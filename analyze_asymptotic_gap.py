"""Asymptotic gap analysis for probe R² between model conditions.

This script analyzes whether the probe R² advantage of embedded models
persists to convergence by fitting asymptotic curves and bootstrapping
confidence intervals.

Usage:
    python analyze_asymptotic_gap.py [--bootstrap N] [--layer LAYER]

The script reads from results/histories.json and saves plots to plots/.

By default, analyzes layer 2 (the embedded layer where small model weights are copied).
Use --layer -1 for post-final-LN layer if desired.
"""

import json
import argparse
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
                                       maxfev=5000)
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
        popt, _ = curve_fit(model, steps, gap, p0=p0, bounds=bounds, maxfev=5000)
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
        popt, _ = curve_fit(model, steps, gap, p0=p0, bounds=bounds, maxfev=5000)
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
                    popt, _ = curve_fit(model, steps, gap, p0=p0, bounds=bounds, maxfev=5000)

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
        'Double power law': fit_double_power_law,
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

def bootstrap_gap_inf(steps, gaps_all_runs, n_bootstrap=5000, verbose=True):
    """Bootstrap confidence interval for asymptotic gap.

    Procedure:
    1. Fit all models to mean trajectory, select best by BIC
    2. For each bootstrap iteration:
       - For each probe step, randomly choose one of the runs
       - Build virtual trajectory from these choices
       - Fit the SAME model (chosen in step 1), extract gap_inf
    3. Compute CI from distribution of gap_inf values

    Args:
        steps: (n_steps,) array of probe steps
        gaps_all_runs: (n_runs, n_steps) array of gap values
        n_bootstrap: number of bootstrap iterations
        verbose: print progress

    Returns:
        dict with gap_inf, ci_95, bootstrap_samples, best_model, etc.
    """
    steps = np.array(steps, dtype=float)
    gaps_all_runs = np.array(gaps_all_runs)
    n_runs, n_steps = gaps_all_runs.shape

    # Step 1: Fit all models to mean trajectory, select best by BIC
    mean_gap = np.mean(gaps_all_runs, axis=0)
    point_fit = select_best_model(steps, mean_gap)

    if point_fit is None:
        print("Warning: Could not fit any model to mean trajectory")
        return {'success': False}

    chosen_model = point_fit['name']
    if verbose:
        print(f"  Selected model: {chosen_model} (BIC: {point_fit['bic']:.1f})")

    # Get the fitting function for the chosen model
    fit_func_map = {
        'Double exponential': fit_double_exp,
        'Single exponential': fit_single_exp,
        'Power law': fit_power_law,
        'Double power law': fit_double_power_law,
    }
    fit_func = fit_func_map[chosen_model]

    # Step 2: Bootstrap using the chosen model
    bootstrap_gap_inf = []
    n_failures = 0

    if verbose:
        print(f"  Running {n_bootstrap} bootstrap iterations...")

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 1000 == 0:
            print(f"    {b + 1}/{n_bootstrap}")

        # For each step, randomly choose one run
        run_choices = np.random.randint(0, n_runs, size=n_steps)
        virtual_gap = gaps_all_runs[run_choices, np.arange(n_steps)]

        # Fit the SAME model we chose earlier
        result = fit_func(steps, virtual_gap)

        if result['success']:
            bootstrap_gap_inf.append(result['gap_inf'])
        else:
            # Fallback: use last few values as estimate
            bootstrap_gap_inf.append(np.mean(virtual_gap[-5:]))
            n_failures += 1

    if verbose and n_failures > 0:
        print(f"  Warning: {n_failures}/{n_bootstrap} fits failed, used fallback")

    bootstrap_gap_inf = np.array(bootstrap_gap_inf)

    # Step 3: Compute statistics
    ci_95 = (np.percentile(bootstrap_gap_inf, 2.5),
             np.percentile(bootstrap_gap_inf, 97.5))
    ci_90 = (np.percentile(bootstrap_gap_inf, 5),
             np.percentile(bootstrap_gap_inf, 95))

    # Fraction positive
    frac_positive = np.mean(bootstrap_gap_inf > 0)

    return {
        'success': True,
        'gap_inf': point_fit['gap_inf'],
        'ci_95': ci_95,
        'ci_90': ci_90,
        'se': np.std(bootstrap_gap_inf),
        'bootstrap_samples': bootstrap_gap_inf,
        'frac_positive': frac_positive,
        'point_fit': point_fit,
        'chosen_model': chosen_model,
        'n_fit_failures': n_failures
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
        'Double power law': fit_double_power_law,
    }

    # Bootstrap each successful model
    per_model_results = {}

    for model_name in successful_models:
        if verbose:
            print(f"  Bootstrapping {model_name}...")

        fit_func = fit_func_map[model_name]
        bootstrap_samples = []
        n_failures = 0

        for b in range(n_bootstrap):
            if verbose and (b + 1) % 2000 == 0:
                print(f"    {b + 1}/{n_bootstrap}")

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

        if verbose and n_failures > 0:
            print(f"    {n_failures}/{n_bootstrap} fits failed")

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
    """Test if probe R² drops when sector indices are shuffled.

    Null hypothesis: Sector targets have no relationship to activations.
    Under H0, shuffling which 4x4 region gets which target value shouldn't
    systematically reduce R².

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

    # True R²
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, Y_train)
    true_r2 = probe.score(X_val, Y_val)

    if verbose:
        print(f"  True R²: {true_r2:.4f}")
        print(f"  Running {n_permutations} permutations...")

    # Permutation test: shuffle sector indices (not samples!)
    null_r2 = []
    for p in range(n_permutations):
        if verbose and (p + 1) % 200 == 0:
            print(f"    {p + 1}/{n_permutations}")

        # Permute the 16 sector columns
        perm = np.random.permutation(16)
        Y_train_perm = Y_train[:, perm]
        Y_val_perm = Y_val[:, perm]

        probe = Ridge(alpha=alpha)
        probe.fit(X_train, Y_train_perm)
        r2 = probe.score(X_val, Y_val_perm)
        null_r2.append(r2)

    null_r2 = np.array(null_r2)
    p_value = np.mean(null_r2 >= true_r2)

    if verbose:
        print(f"\n  Results:")
        print(f"    True R²: {true_r2:.4f}")
        print(f"    Null R² mean: {np.mean(null_r2):.4f} (std: {np.std(null_r2):.4f})")
        print(f"    p-value: {p_value:.4f}")
        if p_value < 0.01:
            print(f"    --> SIGNIFICANT: Sector structure matters (p < 0.01)")
        else:
            print(f"    --> NOT significant: Sector structure may not matter")

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
            edgecolor='white', label=f'Null distribution (n={len(null_dist)})')

    ax.axvline(true_r2, color='red', linewidth=2, linestyle='-',
               label=f'True R² = {true_r2:.4f}')
    ax.axvline(np.mean(null_dist), color='orange', linewidth=2, linestyle='--',
               label=f'Null mean = {np.mean(null_dist):.4f}')

    # p-value annotation
    p_val = result['p_value']
    if p_val < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p_val:.3f}"

    sig_text = "SIGNIFICANT" if p_val < 0.01 else "Not significant"
    color = 'green' if p_val < 0.01 else 'orange'

    ax.text(0.95, 0.95, f"{p_text}\n{sig_text}", transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    ax.set_xlabel('Probe R²', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Permutation Test (Layer {result["layer"]})', fontsize=12)
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

    # Panel 1: Layer profiles
    ax = axes[0]
    for model_type, profile in results['layer_profiles'].items():
        layers = sorted(profile.keys(), key=lambda x: (x < 0, x))
        r2_values = [profile[l] for l in layers]
        x_labels = [str(l) if l >= 0 else 'post-LN' for l in layers]

        ax.plot(range(len(layers)), r2_values, 'o-', color=model_colors.get(model_type, 'gray'),
                linewidth=2, markersize=8, label=model_type)

        # Add error bars if we have seed data
        if model_type in results['seed_data']:
            seed_data = results['seed_data'][model_type]
            stds = [np.std(seed_data.get(l, [0])) for l in layers]
            ax.fill_between(range(len(layers)),
                           np.array(r2_values) - np.array(stds),
                           np.array(r2_values) + np.array(stds),
                           alpha=0.2, color=model_colors.get(model_type, 'gray'))

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(x_labels)
    ax.axvline(2, color='red', linestyle='--', alpha=0.5, label='Layer 2 (embedded)')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Final Probe R²', fontsize=11)
    ax.set_title('Probe R² by Layer', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

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
        'Double power law': '#984ea3',
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
        n_bootstrap: Number of bootstrap iterations
        plots_dir: Directory for output plots
        all_models: If True, bootstrap ALL 4 models and check agreement
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
        result = bootstrap_gap_inf(steps, gaps, n_bootstrap, verbose=True)

        if not result['success']:
            print("  Bootstrap failed")
            return None

        # Report
        print(f"\n  === Results ===")
        print(f"  Best model: {result['point_fit']['name']}")
        print(f"  gap_inf: {result['gap_inf']:.4f} (SE: {result['se']:.4f})")
        print(f"  95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")

        if result['ci_95'][0] > 0:
            print(f"  --> CI EXCLUDES ZERO: Evidence for persistent advantage")
        elif result['ci_95'][1] < 0:
            print(f"  --> CI EXCLUDES ZERO: Evidence for persistent disadvantage")
        else:
            print(f"  --> CI includes zero ({100*result['frac_positive']:.0f}% of bootstrap positive)")

        # Fit quality
        if result['n_fit_failures'] > 0:
            print(f"\n  Note: {result['n_fit_failures']}/{n_bootstrap} bootstrap fits failed (used fallback)")

        # Plot
        output_path = Path(plots_dir) / f"asymptotic_gap_{cond_a.replace('+', '_').replace('(', '').replace(')', '')}_vs_{cond_b.replace('+', '_').replace('(', '').replace(')', '')}.png"
        plot_analysis(steps, gaps, result, label, output_path)

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


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive post-experiment analysis for embedding study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run asymptotic gap analysis (default)
  python analyze_asymptotic_gap.py

  # Run with all models bootstrapped
  python analyze_asymptotic_gap.py --all-models

  # Run alpha sweep analysis
  python analyze_asymptotic_gap.py --alpha-sweep

  # Run permutation test
  python analyze_asymptotic_gap.py --permutation

  # Run layer comparison analysis
  python analyze_asymptotic_gap.py --layer-comparison

  # Run everything
  python analyze_asymptotic_gap.py --full
        """
    )

    # Analysis selection
    analysis = parser.add_argument_group('Analysis Selection')
    analysis.add_argument('--full', action='store_true',
                         help='Run all analyses')
    analysis.add_argument('--asymptotic', action='store_true',
                         help='Run asymptotic gap analysis (default if nothing specified)')
    analysis.add_argument('--alpha-sweep', action='store_true',
                         help='Run alpha sweep analysis on checkpoints')
    analysis.add_argument('--permutation', action='store_true',
                         help='Run permutation test for sector validity')
    analysis.add_argument('--layer-comparison', action='store_true',
                         help='Run multi-layer comparison analysis')

    # Asymptotic options
    asymp = parser.add_argument_group('Asymptotic Analysis Options')
    asymp.add_argument('--all-models', action='store_true',
                      help='Bootstrap all 4 models, not just BIC-selected')
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

    # Default to asymptotic analysis if nothing specified
    if not any([args.full, args.asymptotic, args.alpha_sweep, args.permutation, args.layer_comparison]):
        args.asymptotic = True

    # If --full, enable all
    if args.full:
        args.asymptotic = True
        args.alpha_sweep = True
        args.permutation = True
        args.layer_comparison = True

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
        if args.all_models:
            print("Mode: Bootstrap ALL models and check agreement")
        else:
            print("Mode: Bootstrap BIC-best model only")

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
    # Final Summary
    # =========================================================================
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print("="*80)
    print(f"Analyses run: {', '.join(analyses_run)}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
