"""Dry run of the full experiment pipeline.

Runs every function in the correct order with minimal data/steps to catch
errors before the long experiment run. Should complete in ~5 minutes.
"""

import os
import sys
import time
import shutil
import numpy as np
import torch

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SMALL_CONFIG, LARGE_CONFIG
from model import create_small_model, create_large_model, count_parameters
from data_loader import EfficientDomineeringDataset
from training import train_for_steps, collate_batch, evaluate, compute_losses
from embedding import embed_small_into_large, verify_embedding
from probing import train_probes_all_layers
from run_experiment import (
    merge_npz_files, quick_validate, train_with_probes,
    update_plots_incrementally, update_probe_plot_incrementally
)
import domineering_game as dg


def generate_synthetic_games(n_games=100, output_path='data/dry_run_games.npz'):
    """Generate a small synthetic dataset for testing."""
    print(f"  Generating {n_games} synthetic games...")

    all_moves = []
    all_lengths = []
    all_winners = []

    np.random.seed(42)

    for _ in range(n_games):
        game = dg.domineering_game()
        moves = []

        while not game[3]:  # while not game_over
            legal = dg.legal_moves(game)
            legal_indices = np.where(legal)[0]
            if len(legal_indices) == 0:
                break
            move = np.random.choice(legal_indices)
            moves.append(move)
            dg.make_move(game, move)

        all_moves.append(moves)
        all_lengths.append(len(moves))
        # Winner based on who made the last move (other player has no moves)
        all_winners.append(len(moves) % 2 == 1)  # Vertical wins if odd moves

    # Pad to same length
    max_len = max(all_lengths)
    padded_moves = np.full((n_games, max_len), -1, dtype=np.int16)
    for i, moves in enumerate(all_moves):
        padded_moves[i, :len(moves)] = moves

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        moves=padded_moves,
        lengths=np.array(all_lengths, dtype=np.int16),
        winners=np.array(all_winners, dtype=bool)
    )
    print(f"  Saved to {output_path}")
    return output_path


def run_dry_run():
    """Run the full experiment pipeline with minimal data."""

    print("=" * 70)
    print("DRY RUN - Testing full experiment pipeline")
    print("=" * 70)

    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create temporary directories
    dry_run_dir = 'dry_run_output'
    os.makedirs(dry_run_dir, exist_ok=True)
    data_dir = os.path.join(dry_run_dir, 'data')
    checkpoint_dir = os.path.join(dry_run_dir, 'checkpoints')
    plots_dir = os.path.join(dry_run_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    errors = []

    # =========================================================================
    # Step 1: Generate synthetic data
    # =========================================================================
    print("\n[1/8] Generating synthetic data...")
    try:
        game_path = generate_synthetic_games(
            n_games=100,
            output_path=os.path.join(data_dir, 'synthetic_games.npz')
        )
        print("  PASS")
    except Exception as e:
        errors.append(f"Data generation: {e}")
        print(f"  FAIL: {e}")
        return errors

    # =========================================================================
    # Step 2: Test data loading and merging
    # =========================================================================
    print("\n[2/8] Testing data loading and merging...")
    try:
        # Test merge (even with single file)
        combined_path = os.path.join(data_dir, 'combined.npz')
        merge_npz_files([game_path], combined_path)

        # Test dataset creation
        train_dataset = EfficientDomineeringDataset(combined_path, split='train', positions_per_game=10)
        val_dataset = EfficientDomineeringDataset(combined_path, split='val', positions_per_game=5)

        # Precompute epoch positions
        train_dataset.precompute_epoch()
        val_dataset.precompute_epoch()

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False,
                                  num_workers=0, collate_fn=collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                num_workers=0, collate_fn=collate_batch)

        # Verify we can iterate
        batch = next(iter(train_loader))
        assert batch['tokens'].shape[1] == 257, "Wrong token shape"
        assert batch['mask'].shape[1] == 480, "Wrong mask shape"
        print(f"  Train: {len(train_dataset)} positions, Val: {len(val_dataset)} positions")
        print("  PASS")
    except Exception as e:
        errors.append(f"Data loading: {e}")
        print(f"  FAIL: {e}")
        return errors

    # =========================================================================
    # Step 3: Test model creation
    # =========================================================================
    print("\n[3/8] Testing model creation...")
    try:
        small_model = create_small_model()
        large_model = create_large_model()
        print(f"  Small: {count_parameters(small_model):,} params")
        print(f"  Large: {count_parameters(large_model):,} params")

        # Test forward pass
        tokens = batch['tokens'].to(device)
        mask = batch['mask'].to(device)
        small_model.to(device)
        large_model.to(device)

        with torch.no_grad():
            v, p, s = small_model(tokens, mask)
            assert v.shape == (32, 1), f"Value shape: {v.shape}"
            assert p.shape == (32, 480), f"Policy shape: {p.shape}"
            assert s.shape == (32, 16), f"Sector shape: {s.shape}"

            v, p, s = large_model(tokens, mask)
            assert v.shape == (32, 1), f"Value shape: {v.shape}"

        print("  PASS")
    except Exception as e:
        errors.append(f"Model creation: {e}")
        print(f"  FAIL: {e}")
        return errors

    # =========================================================================
    # Step 4: Test training loop with probes (small model, 1 epoch)
    # =========================================================================
    print("\n[4/8] Testing training loop with periodic probing/validation...")
    try:
        # Use short intervals to trigger probing and validation multiple times
        # With ~80 train positions / 32 batch_size ≈ 2-3 steps per epoch
        # So we use very short intervals to ensure they trigger
        small_aux = create_small_model()
        all_histories = {}

        hist = train_with_probes(
            small_aux, train_loader, val_loader,
            n_epochs=1,
            model_name='small_aux_test',
            use_auxiliary=True,
            device=device,
            probe_interval=2,        # Probe every 2 steps
            quick_val_interval=1,    # Validate every step
            train_log_interval=1,    # Log every step
            checkpoint_dir=checkpoint_dir,
            all_histories=all_histories,
            plots_dir=plots_dir
        )

        print(f"  Steps logged: {len(hist['step_train_loss'])} train, {len(hist['step_val_loss'])} val")
        print(f"  Probe evaluations: {len(hist['probe_steps'])}")
        print(f"  Final val loss: {hist['val_loss'][-1]:.4f}")

        # Small models skip step-level validation and probing (only large models matter)
        # Just verify training loop ran successfully with epoch-level val
        assert len(hist['val_loss']) >= 1, "No epoch-level validation done"

        # Save checkpoint for embedding test
        torch.save(
            {'model_state_dict': small_aux.state_dict()},
            os.path.join(checkpoint_dir, 'small_aux_test.pt')
        )
        print("  PASS")
    except Exception as e:
        errors.append(f"Training loop: {e}")
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return errors

    # =========================================================================
    # Step 5: Test embedding
    # =========================================================================
    print("\n[5/8] Testing embedding...")
    try:
        small_trained = create_small_model()
        small_trained.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'small_aux_test.pt'),
                      weights_only=True)['model_state_dict']
        )

        large_embedded = create_large_model()
        embed_small_into_large(small_trained, large_embedded)

        # Verify embedding
        test_tokens = torch.randint(0, 2, (4, 257))
        test_tokens[:, -1] = 2
        result = verify_embedding(small_trained, large_embedded, test_tokens)

        max_diff = max(result['value_diff'], result['policy_diff'], result['sector_diff'])
        print(f"  Max embedding diff: {max_diff:.2e}")
        assert max_diff < 1e-5, f"Embedding verification failed: {max_diff}"
        print("  PASS")
    except Exception as e:
        errors.append(f"Embedding: {e}")
        print(f"  FAIL: {e}")
        return errors

    # =========================================================================
    # Step 6: Test probing
    # =========================================================================
    print("\n[6/8] Testing probing...")
    try:
        large_embedded.to(device)
        probes = train_probes_all_layers(large_embedded, val_loader, n_samples=100, device=device)

        print(f"  Probed {len(probes)} layers")
        for layer_idx, probe in sorted(probes.items()):
            print(f"    Layer {layer_idx}: R² = {probe.val_r2:.4f}")
        print("  PASS")
    except Exception as e:
        errors.append(f"Probing: {e}")
        print(f"  FAIL: {e}")
        return errors

    # =========================================================================
    # Step 7: Test quick_validate with step-based sampling
    # =========================================================================
    print("\n[7/8] Testing quick_validate with step-based sampling...")
    try:
        loss1 = quick_validate(large_embedded, val_loader, device, step=500)
        loss2 = quick_validate(large_embedded, val_loader, device, step=500)
        loss3 = quick_validate(large_embedded, val_loader, device, step=1000)

        print(f"  Step 500 (run 1): {loss1:.4f}")
        print(f"  Step 500 (run 2): {loss2:.4f}")
        print(f"  Step 1000: {loss3:.4f}")
        print("  PASS")
    except Exception as e:
        errors.append(f"Quick validate: {e}")
        print(f"  FAIL: {e}")
        return errors

    # =========================================================================
    # Step 8: Test large model training with plot updates
    # =========================================================================
    print("\n[8/10] Testing large model training with plot updates...")
    try:
        all_histories = {}

        # Train the embedded large model briefly
        large_for_test = create_large_model()
        embed_small_into_large(small_trained, large_for_test)

        hist = train_with_probes(
            large_for_test, train_loader, val_loader,
            n_epochs=1,
            model_name='large_embed_aux_test',  # Contains 'large' so plots will update
            use_auxiliary=False,
            device=device,
            probe_interval=2,
            quick_val_interval=1,
            train_log_interval=1,
            checkpoint_dir=checkpoint_dir,
            all_histories=all_histories,
            plots_dir=plots_dir
        )
        all_histories['Large+embed(aux)_s42'] = hist

        print(f"  Steps logged: {len(hist['step_train_loss'])} train, {len(hist['step_val_loss'])} val")
        print(f"  Probe evaluations: {len(hist['probe_steps'])}")

        # Check that plot files were created
        assert os.path.exists(os.path.join(plots_dir, 'loss_curves_live.png')), "Loss curves not generated"
        assert os.path.exists(os.path.join(plots_dir, 'probe_r2_live.png')), "Probe plot not generated"
        print("  PASS")
    except Exception as e:
        errors.append(f"Large model training: {e}")
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return errors

    # =========================================================================
    # Step 9: Test plot generation with mock multi-seed data
    # =========================================================================
    print("\n[9/10] Testing plot generation with multi-seed data...")
    try:
        from collections import defaultdict

        # Create mock histories
        mock_histories = {
            'Large-baseline_s42': {
                'step_train_loss': [(100, 1.5), (200, 1.3), (300, 1.1)],
                'step_val_loss': [(100, 1.6), (200, 1.4), (300, 1.2)],
                'probe_steps': [100, 200, 300],
                'probe_r2': {2: [0.3, 0.4, 0.5]},
            },
            'Large+embed(aux)_s42': {
                'step_train_loss': [(100, 1.2), (200, 1.0), (300, 0.9)],
                'step_val_loss': [(100, 1.3), (200, 1.1), (300, 1.0)],
                'probe_steps': [100, 200, 300],
                'probe_r2': {2: [0.5, 0.6, 0.7]},
            },
            'Large+embed(noaux)_s42': {
                'step_train_loss': [(100, 1.3), (200, 1.1), (300, 0.95)],
                'step_val_loss': [(100, 1.4), (200, 1.2), (300, 1.05)],
                'probe_steps': [100, 200, 300],
                'probe_r2': {2: [0.4, 0.5, 0.55]},
            },
        }

        update_plots_incrementally(mock_histories, plots_dir)
        update_probe_plot_incrementally(mock_histories, plots_dir)

        # Check files exist
        assert os.path.exists(os.path.join(plots_dir, 'loss_curves_live.png'))
        assert os.path.exists(os.path.join(plots_dir, 'probe_r2_live.png'))
        print(f"  Plots saved to {plots_dir}/")
        print("  PASS")
    except Exception as e:
        errors.append(f"Plot generation: {e}")
        print(f"  FAIL: {e}")
        return errors

    # =========================================================================
    # Step 10: Test full single-seed run structure
    # =========================================================================
    print("\n[10/10] Testing full single-seed experiment structure...")
    try:
        # This tests that we can run all 5 model types in sequence
        # without errors, using minimal training
        from collections import defaultdict
        all_histories = {}
        seed = 999

        # Training order matches run_experiment.py:
        # 1. Large baseline (for time estimation)
        # 2. Small noaux -> Large embed noaux
        # 3. Small aux -> Large embed aux
        models_to_train = [
            ('large_baseline', False, False),   # (name_suffix, is_small, use_aux)
            ('small_noaux', True, False),
            ('large_embed_noaux', False, False),
            ('small_aux', True, True),
            ('large_embed_aux', False, False),
        ]

        for name_suffix, is_small, use_aux in models_to_train:
            model_name = f'{name_suffix}_s{seed}'
            print(f"    Training {model_name}...")

            if is_small:
                model = create_small_model()
            elif 'embed_aux' in name_suffix:
                model = create_large_model()
                # Load trained small+aux and embed
                small_src = create_small_model()
                small_src.load_state_dict(torch.load(
                    os.path.join(checkpoint_dir, 'small_aux_s999_best.pt'),
                    weights_only=True
                ))
                embed_small_into_large(small_src, model)
            elif 'embed_noaux' in name_suffix:
                model = create_large_model()
                small_src = create_small_model()
                small_src.load_state_dict(torch.load(
                    os.path.join(checkpoint_dir, 'small_noaux_s999_best.pt'),
                    weights_only=True
                ))
                embed_small_into_large(small_src, model)
            else:
                model = create_large_model()

            hist = train_with_probes(
                model, train_loader, val_loader,
                n_epochs=1,
                model_name=model_name,
                use_auxiliary=use_aux,
                device=device,
                probe_interval=5,
                quick_val_interval=2,
                train_log_interval=1,
                checkpoint_dir=checkpoint_dir,
                all_histories=all_histories,
                plots_dir=plots_dir
            )

            # Map to expected history key format
            if 'small_aux' in name_suffix and 'noaux' not in name_suffix:
                all_histories[f'Small+aux_s{seed}'] = hist
            elif 'small_noaux' in name_suffix:
                all_histories[f'Small-noaux_s{seed}'] = hist
            elif 'large_baseline' in name_suffix:
                all_histories[f'Large-baseline_s{seed}'] = hist
            elif 'large_embed_aux' in name_suffix and 'noaux' not in name_suffix:
                all_histories[f'Large+embed(aux)_s{seed}'] = hist
            elif 'large_embed_noaux' in name_suffix:
                all_histories[f'Large+embed(noaux)_s{seed}'] = hist

        print(f"  Trained {len(models_to_train)} models successfully")
        print("  PASS")
    except Exception as e:
        errors.append(f"Full structure test: {e}")
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return errors

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    if errors:
        print(f"DRY RUN FAILED - {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("DRY RUN PASSED - All systems go!")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Output: {dry_run_dir}/")
        print("\nYou can now run the full experiment with confidence.")
    print("=" * 70)

    return errors


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Dry run of experiment pipeline')
    parser.add_argument('--keep-output', action='store_true',
                        help='Keep the dry_run_output directory after completion')
    args = parser.parse_args()

    errors = run_dry_run()

    if not args.keep_output and not errors:
        print("\nCleaning up dry_run_output/...")
        shutil.rmtree('dry_run_output', ignore_errors=True)
        print("Done.")

    sys.exit(1 if errors else 0)
