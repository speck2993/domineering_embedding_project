"""Test bootstrapped gap_inf estimates with full vs truncated data."""

import numpy as np
from analyze_experiment import fit_single_exp, fit_double_exp, fit_power_law

def bootstrap_gap_inf(steps, gaps, fit_func, n_bootstrap=300, truncate_frac=None):
    """Bootstrap gap_inf estimates, optionally truncating data first.

    Args:
        steps: (n_steps,) array of training steps
        gaps: (n_runs, n_steps) array of gap values
        fit_func: Fitting function to use
        n_bootstrap: Number of bootstrap iterations
        truncate_frac: If provided, use only first X fraction of data

    Returns:
        dict with gap_inf, ci_95, point estimate success
    """
    if truncate_frac is not None:
        n_keep = int(len(steps) * truncate_frac)
        steps = steps[:n_keep]
        gaps = gaps[:, :n_keep]

    # Point estimate on mean
    mean_gap = np.mean(gaps, axis=0)
    point_fit = fit_func(steps, mean_gap)

    if not point_fit['success']:
        return {
            'gap_inf': np.nan,
            'ci_95': (np.nan, np.nan),
            'success': False,
            'n_points': len(steps)
        }

    # Bootstrap
    n_runs = gaps.shape[0]
    bootstrap_samples = []

    for _ in range(n_bootstrap):
        # Resample runs with replacement
        sample_idx = np.random.choice(n_runs, size=n_runs, replace=True)
        sample_gaps = gaps[sample_idx, :]
        sample_mean = np.mean(sample_gaps, axis=0)

        # Fit model
        fit_result = fit_func(steps, sample_mean)
        if fit_result['success']:
            bootstrap_samples.append(fit_result['gap_inf'])

    if len(bootstrap_samples) < 0.9 * n_bootstrap:
        return {
            'gap_inf': point_fit['gap_inf'],
            'ci_95': (np.nan, np.nan),
            'success': False,
            'n_points': len(steps)
        }

    ci_95 = (np.percentile(bootstrap_samples, 2.5), np.percentile(bootstrap_samples, 97.5))

    return {
        'gap_inf': point_fit['gap_inf'],
        'ci_95': ci_95,
        'se': np.std(bootstrap_samples),
        'success': True,
        'n_points': len(steps)
    }


# Generate synthetic data with noisy tail
n_runs = 5
n_steps = 61
steps = np.linspace(1000, 61000, n_steps)
gaps = np.zeros((n_runs, n_steps))

# True model: power law with gap_inf = 0.02
# Add extra noise to last 20%
for i in range(n_runs):
    base = 0.02 + 0.15 * (steps + 1000)**(-0.5)
    noise = np.random.randn(n_steps) * 0.005

    # Add extra noise to last 20%
    noise_boost = np.zeros(n_steps)
    last_20_idx = int(n_steps * 0.8)
    noise_boost[last_20_idx:] = np.random.randn(n_steps - last_20_idx) * 0.010

    gaps[i, :] = base + noise + noise_boost

print("Testing bootstrapped gap_inf estimates with noisy tail")
print("="*70)
print("True model: gap(t) = 0.02 + 0.15*(t+1000)^(-0.5)")
print("Noise: ±0.005 for first 80%, ±0.015 for last 20%")
print()

models = {
    'Single exponential': fit_single_exp,
    'Double exponential': fit_double_exp,
    'Power law': fit_power_law,
}

for model_name, fit_func in models.items():
    print(f"\n{model_name}:")
    print("-" * 70)

    # Full data
    result_full = bootstrap_gap_inf(steps, gaps, fit_func, n_bootstrap=300)

    # Truncated to 80%
    result_80 = bootstrap_gap_inf(steps, gaps, fit_func, n_bootstrap=300, truncate_frac=0.8)

    print(f"  Full data ({result_full['n_points']} points):")
    if result_full['success']:
        print(f"    gap_inf = {result_full['gap_inf']:.5f}")
        print(f"    95% CI  = [{result_full['ci_95'][0]:.5f}, {result_full['ci_95'][1]:.5f}]")
        print(f"    SE      = {result_full['se']:.5f}")
        ci_width_full = result_full['ci_95'][1] - result_full['ci_95'][0]
        print(f"    CI width = {ci_width_full:.5f}")
    else:
        print(f"    FAILED")

    print(f"\n  Truncated 80% ({result_80['n_points']} points):")
    if result_80['success']:
        print(f"    gap_inf = {result_80['gap_inf']:.5f}")
        print(f"    95% CI  = [{result_80['ci_95'][0]:.5f}, {result_80['ci_95'][1]:.5f}]")
        print(f"    SE      = {result_80['se']:.5f}")
        ci_width_80 = result_80['ci_95'][1] - result_80['ci_95'][0]
        print(f"    CI width = {ci_width_80:.5f}")

        if result_full['success']:
            print(f"\n  Comparison:")
            print(f"    Δ gap_inf = {result_80['gap_inf'] - result_full['gap_inf']:.5f}")
            print(f"    CI width reduction = {(1 - ci_width_80/ci_width_full)*100:.1f}%")
    else:
        print(f"    FAILED")

print("\n" + "="*70)
print("Summary: Does truncating to 80% reduce uncertainty?")
print("="*70)
