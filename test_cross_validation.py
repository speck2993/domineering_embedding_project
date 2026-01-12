"""Test cross-validation function with synthetic data."""

import numpy as np
from analyze_experiment import cross_validate_models, fit_single_exp, fit_power_law

# Generate synthetic data with power-law decay
n_runs = 3
n_steps = 61
steps = np.linspace(1000, 61000, n_steps)
gaps = np.zeros((n_runs, n_steps))

# True model: power law with gap_inf = 0.02
for i in range(n_runs):
    gaps[i, :] = 0.02 + 0.15 * (steps + 1000)**(-0.5) + np.random.randn(n_steps) * 0.005

print("Testing cross-validation with synthetic power-law data")
print("="*70)
print("True model: gap(t) = 0.02 + 0.15*(t+1000)^(-0.5) + noise")
print()

# Run cross-validation
cv_results = cross_validate_models(steps, gaps, train_fractions=[0.2, 0.4, 0.6, 0.8], verbose=True)

print("\n" + "="*70)
print("Expected: Power law should have lowest test RMSE")
print(f"Actual best model: {cv_results['best_model']}")
print("="*70)

if cv_results['best_model'] == 'Power law':
    print("✓ TEST PASSED: Power law correctly identified as best model")
else:
    print("✗ TEST FAILED: Expected power law, but got", cv_results['best_model'])
