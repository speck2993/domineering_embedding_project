"""Test autocorrelation diagnostic functions with synthetic data."""

import numpy as np
import sys

# Import the functions from analyze_experiment
from analyze_experiment import (
    compute_within_acf,
    compute_between_ccf,
    test_autocorrelation_structure
)


def test_ar1_high_autocorrelation():
    """Test detection of AR(1) process with high autocorrelation."""
    print("="*70)
    print("Test 1: AR(1) Process with rho=0.8 (High Autocorrelation)")
    print("="*70)

    # Generate AR(1) with rho=0.8
    n_runs, n_steps = 3, 30
    rho = 0.8
    gaps = np.zeros((n_runs, n_steps))

    for i in range(n_runs):
        gaps[i, 0] = np.random.randn()
        for t in range(1, n_steps):
            gaps[i, t] = rho * gaps[i, t-1] + np.random.randn() * 0.5

    steps = np.arange(1000, 1000 + n_steps * 1000, 1000)

    # Run test
    result = test_autocorrelation_structure(steps, gaps, max_lag=3, n_permutations=500, verbose=False)

    # Check results
    print(f"\nResults:")
    print(f"  Within-seed ACF[1]: {result['within_seed_acf'][0]:.3f}")
    print(f"  Threat level: {result['threat_level']}")
    print(f"  VIF: {result['vif']:.2f}")
    print(f"  P-value[1]: {result['p_values'][0]:.3f}")

    # Assertions
    assert result['within_seed_acf'][0] > 0.5, f"Should detect high ACF, got {result['within_seed_acf'][0]}"
    assert result['threat_level'] in ['moderate', 'high'], f"Should flag threat, got {result['threat_level']}"
    assert result['vif'] > 1.3, f"Should show variance inflation, got VIF={result['vif']}"

    print("\n✓ AR(1) test PASSED")
    return True


def test_iid_low_autocorrelation():
    """Test that IID data is classified as low threat."""
    print("\n" + "="*70)
    print("Test 2: IID Random Data (No Autocorrelation)")
    print("="*70)

    n_runs, n_steps = 3, 30
    gaps = np.random.randn(n_runs, n_steps)
    steps = np.arange(1000, 1000 + n_steps * 1000, 1000)

    # Run test
    result = test_autocorrelation_structure(steps, gaps, max_lag=3, n_permutations=500, verbose=False)

    # Check results
    print(f"\nResults:")
    print(f"  Within-seed ACF[1]: {result['within_seed_acf'][0]:.3f}")
    print(f"  Threat level: {result['threat_level']}")
    print(f"  VIF: {result['vif']:.2f}")
    print(f"  P-value[1]: {result['p_values'][0]:.3f}")

    # Assertions
    assert abs(result['within_seed_acf'][0]) < 0.5, f"Should have low ACF, got {result['within_seed_acf'][0]}"
    assert result['threat_level'] == 'low', f"Should be low threat, got {result['threat_level']}"
    assert result['vif'] < 1.7, f"Should not inflate variance much, got VIF={result['vif']}"

    print("\n✓ IID test PASSED")
    return True


def test_helper_functions():
    """Test that helper functions work correctly."""
    print("\n" + "="*70)
    print("Test 3: Helper Functions")
    print("="*70)

    # Simple test data
    n_runs, n_steps = 3, 20
    gaps = np.random.randn(n_runs, n_steps)

    # Test compute_within_acf
    within_acf = compute_within_acf(gaps, max_lag=3)
    assert within_acf.shape == (3,), f"Wrong ACF shape: {within_acf.shape}"
    assert not np.any(np.isnan(within_acf)), "ACF contains NaN"
    print(f"  Within-seed ACF: {within_acf}")

    # Test compute_between_ccf
    between_ccf = compute_between_ccf(gaps, max_lag=3)
    assert between_ccf.shape == (3,), f"Wrong CCF shape: {between_ccf.shape}"
    assert not np.any(np.isnan(between_ccf)), "CCF contains NaN"
    print(f"  Between-seed CCF: {between_ccf}")

    print("\n✓ Helper functions test PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\nRunning Autocorrelation Diagnostic Tests")
    print("="*70)

    tests = [
        test_helper_functions,
        test_iid_low_autocorrelation,
        test_ar1_high_autocorrelation,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ TEST ERROR: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
