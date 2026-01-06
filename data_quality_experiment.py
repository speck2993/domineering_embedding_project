"""Data quality validation experiment.

Validates that self-play data improves model quality compared to alpha-beta alone.

Trains 3 medium models:
- Model A: phase0 only, 3 epochs (baseline)
- Model B: combined data, same STEPS as A (per-sample quality test)
- Model C: combined data, 3 epochs (full data test)

Evaluates via validation loss and head-to-head games.
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE
from model import create_medium_model, count_parameters
from data_loader import DomineeringDataset
from training import train_model, train_for_steps, evaluate, collate_batch
from selfplay import compare_models


# ============================================================================
# Data Utilities
# ============================================================================

def merge_npz_files(paths, output_path):
    """Merge multiple NPZ game files into one.

    Args:
        paths: List of paths to NPZ files
        output_path: Path for merged output

    Returns:
        Dict with merge statistics
    """
    all_moves = []
    all_lengths = []
    all_winners = []

    for path in paths:
        data = np.load(path)
        all_moves.append(data['moves'])
        all_lengths.append(data['lengths'])
        all_winners.append(data['winners'])
        print(f"  Loaded {path}: {len(data['moves'])} games")

    # Handle different max_lengths by padding
    max_len = max(m.shape[1] for m in all_moves)
    padded_moves = []
    for m in all_moves:
        if m.shape[1] < max_len:
            pad_width = ((0, 0), (0, max_len - m.shape[1]))
            m = np.pad(m, pad_width, constant_values=-1)
        padded_moves.append(m)

    moves = np.concatenate(padded_moves, axis=0)
    lengths = np.concatenate(all_lengths)
    winners = np.concatenate(all_winners)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.savez_compressed(output_path, moves=moves, lengths=lengths, winners=winners)

    total_positions = int(lengths.sum())
    print(f"  Merged: {len(moves)} games, ~{total_positions:,} positions -> {output_path}")

    return {
        'n_games': len(moves),
        'total_positions': total_positions,
        'path': output_path
    }


def get_dataset_stats(npz_path):
    """Get basic statistics for a dataset."""
    data = np.load(npz_path)
    n_games = len(data['moves'])
    total_positions = int(data['lengths'].sum())
    return {'n_games': n_games, 'total_positions': total_positions}


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(phase0_path, selfplay_path, n_epochs=3, n_openings=200,
                   device='cuda', seed=42):
    """Run the full data quality experiment.

    Args:
        phase0_path: Path to phase0_games.npz
        selfplay_path: Path to selfplay_games.npz
        n_epochs: Number of epochs for Models A and C
        n_openings: Number of openings for head-to-head (total games = 2x)
        device: Device for training
        seed: Random seed

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("DATA QUALITY VALIDATION EXPERIMENT")
    print("=" * 60)

    start_time = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Prepare data
    # -------------------------------------------------------------------------
    print("\n[1/5] Preparing data...")

    phase0_stats = get_dataset_stats(phase0_path)
    selfplay_stats = get_dataset_stats(selfplay_path)

    print(f"  Phase0: {phase0_stats['n_games']} games, ~{phase0_stats['total_positions']:,} positions")
    print(f"  Selfplay: {selfplay_stats['n_games']} games, ~{selfplay_stats['total_positions']:,} positions")

    # Create combined dataset
    combined_path = 'data/combined_games.npz'
    print("\n  Merging datasets...")
    merge_npz_files([phase0_path, selfplay_path], combined_path)

    # -------------------------------------------------------------------------
    # Step 2: Create data loaders
    # -------------------------------------------------------------------------
    print("\n[2/5] Creating data loaders...")

    # Phase0 loaders
    train_dataset_a = DomineeringDataset(phase0_path, split='train')
    val_dataset_a = DomineeringDataset(phase0_path, split='val')

    # Combined loaders
    train_dataset_c = DomineeringDataset(combined_path, split='train')
    val_dataset_c = DomineeringDataset(combined_path, split='val')

    n_workers = 4 if device == 'cuda' else 2
    pin_memory = device == 'cuda'

    train_loader_a = DataLoader(train_dataset_a, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=n_workers, collate_fn=collate_batch,
                                pin_memory=pin_memory)
    val_loader_a = DataLoader(val_dataset_a, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=n_workers, collate_fn=collate_batch,
                              pin_memory=pin_memory)

    train_loader_c = DataLoader(train_dataset_c, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=n_workers, collate_fn=collate_batch,
                                pin_memory=pin_memory)
    val_loader_c = DataLoader(val_dataset_c, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=n_workers, collate_fn=collate_batch,
                              pin_memory=pin_memory)

    # Calculate steps for Model A (3 epochs on phase0)
    steps_a = n_epochs * len(train_loader_a)
    print(f"\n  Model A will train for {steps_a} steps ({n_epochs} epochs on phase0)")
    print(f"  Model B will train for {steps_a} steps (same steps on combined)")
    print(f"  Model C will train for {n_epochs} epochs on combined ({n_epochs * len(train_loader_c)} steps)")

    # -------------------------------------------------------------------------
    # Step 3: Train models
    # -------------------------------------------------------------------------
    print("\n[3/5] Training models...")

    # Model A: phase0 only, 3 epochs
    print("\n--- Model A (phase0, 3 epochs) ---")
    model_a = create_medium_model()
    print(f"  Parameters: {count_parameters(model_a):,}")
    metrics_a = train_model(model_a, train_loader_a, val_loader_a, n_epochs=n_epochs,
                            use_auxiliary=False, device=device, seed=seed)

    # Model B: combined, same steps as A
    print("\n--- Model B (combined, same steps as A) ---")
    model_b = create_medium_model()
    metrics_b = train_for_steps(model_b, train_loader_c, val_loader_c, n_steps=steps_a,
                                use_auxiliary=False, device=device, seed=seed)

    # Model C: combined, 3 epochs
    print("\n--- Model C (combined, 3 epochs) ---")
    model_c = create_medium_model()
    metrics_c = train_model(model_c, train_loader_c, val_loader_c, n_epochs=n_epochs,
                            use_auxiliary=False, device=device, seed=seed)

    # -------------------------------------------------------------------------
    # Step 4: Head-to-head evaluation
    # -------------------------------------------------------------------------
    print("\n[4/5] Head-to-head evaluation...")

    print(f"\n  A vs B ({n_openings} openings)...")
    h2h_ab = compare_models(model_a, model_b, n_openings=n_openings, device=device)
    print(f"    Decisive: A={h2h_ab['decisive_a']}, B={h2h_ab['decisive_b']}, washes={h2h_ab['washes']}, p={h2h_ab['p_value']:.4f}")

    print(f"\n  A vs C ({n_openings} openings)...")
    h2h_ac = compare_models(model_a, model_c, n_openings=n_openings, device=device)
    print(f"    Decisive: A={h2h_ac['decisive_a']}, C={h2h_ac['decisive_b']}, washes={h2h_ac['washes']}, p={h2h_ac['p_value']:.4f}")

    print(f"\n  B vs C ({n_openings} openings)...")
    h2h_bc = compare_models(model_b, model_c, n_openings=n_openings, device=device)
    print(f"    Decisive: B={h2h_bc['decisive_a']}, C={h2h_bc['decisive_b']}, washes={h2h_bc['washes']}, p={h2h_bc['p_value']:.4f}")

    # -------------------------------------------------------------------------
    # Step 5: Results summary
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("[5/5] RESULTS SUMMARY")
    print("=" * 60)

    print("\nVALIDATION METRICS:")
    print("-" * 60)
    print(f"{'Model':<10} {'Loss':>10} {'V-Loss':>10} {'P-Loss':>10} {'V-Acc':>10} {'P-Acc':>10}")
    print("-" * 60)
    for name, metrics in [('A (phase0)', metrics_a), ('B (comb-eq)', metrics_b), ('C (comb-full)', metrics_c)]:
        print(f"{name:<10} {metrics['loss']:>10.4f} {metrics['value_loss']:>10.4f} "
              f"{metrics['policy_loss']:>10.4f} {metrics['value_acc']:>10.1%} {metrics['policy_acc']:>10.1%}")

    print("\nHEAD-TO-HEAD RESULTS (decisive pairs):")
    print("-" * 60)
    print(f"{'Matchup':<10} {'Model 1':>10} {'Model 2':>10} {'Washes':>10} {'p-value':>10}")
    print("-" * 60)
    print(f"{'A vs B':<10} {h2h_ab['decisive_a']:>10} {h2h_ab['decisive_b']:>10} {h2h_ab['washes']:>10} {h2h_ab['p_value']:>10.4f}")
    print(f"{'A vs C':<10} {h2h_ac['decisive_a']:>10} {h2h_ac['decisive_b']:>10} {h2h_ac['washes']:>10} {h2h_ac['p_value']:>10.4f}")
    print(f"{'B vs C':<10} {h2h_bc['decisive_a']:>10} {h2h_bc['decisive_b']:>10} {h2h_bc['washes']:>10} {h2h_bc['p_value']:>10.4f}")

    print("\nINTERPRETATION (p < 0.05 = significant):")
    print("-" * 60)

    # Primary question: Does more data help? (C vs A)
    # Note: In h2h_ac, 'decisive_a' is A's wins, 'decisive_b' is C's wins
    if h2h_ac['decisive_b'] > h2h_ac['decisive_a'] and h2h_ac['p_value'] < 0.05:
        print("✓ C >> A: More data (phase0 + selfplay) produces significantly better models")
    elif h2h_ac['decisive_a'] > h2h_ac['decisive_b'] and h2h_ac['p_value'] < 0.05:
        print("✗ A >> C: More data did NOT help (unexpected!)")
    else:
        print("~ C ≈ A: No significant difference")

    # Secondary question: Is combined data better per-sample? (B vs A)
    if h2h_ab['decisive_b'] > h2h_ab['decisive_a'] and h2h_ab['p_value'] < 0.05:
        print("✓ B >> A: Combined data is higher quality per-sample")
    elif h2h_ab['decisive_a'] > h2h_ab['decisive_b'] and h2h_ab['p_value'] < 0.05:
        print("✗ A >> B: Phase0 data is higher quality per-sample (unexpected!)")
    else:
        print("~ B ≈ A: No significant difference in per-sample quality")

    # Does more training on all data help? (C vs B)
    if h2h_bc['decisive_b'] > h2h_bc['decisive_a'] and h2h_bc['p_value'] < 0.05:
        print("✓ C >> B: Longer training on combined data helps")
    elif h2h_bc['decisive_a'] > h2h_bc['decisive_b'] and h2h_bc['p_value'] < 0.05:
        print("? B >> C: More training hurt (possible overfitting)")
    else:
        print("~ C ≈ B: No significant difference from additional training")

    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Clean up temporary combined file
    if os.path.exists(combined_path):
        os.remove(combined_path)
        print(f"\nCleaned up {combined_path}")

    return {
        'metrics': {'A': metrics_a, 'B': metrics_b, 'C': metrics_c},
        'h2h': {'AB': h2h_ab, 'AC': h2h_ac, 'BC': h2h_bc},
        'elapsed': elapsed
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Data quality validation experiment')
    parser.add_argument('--phase0', type=str, default='data/phase0_games.npz',
                        help='Path to phase0 games')
    parser.add_argument('--selfplay', type=str, default='data/selfplay_games.npz',
                        help='Path to selfplay games')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs for Models A and C')
    parser.add_argument('--openings', type=int, default=200,
                        help='Number of openings for head-to-head (total games = 2x)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    run_experiment(
        phase0_path=args.phase0,
        selfplay_path=args.selfplay,
        n_epochs=args.epochs,
        n_openings=args.openings,
        device=device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
