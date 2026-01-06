"""Dry-run tests for the experiment pipeline.

Tests every component of the experiment without running full training.
Run with: python test_experiment_pipeline.py [--unit | --integration | --all]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import defaultdict

import numpy as np
import torch

# ============================================================================
# Test Configuration
# ============================================================================

TEST_DIR = 'test_artifacts'
TEST_CHECKPOINT_DIR = os.path.join(TEST_DIR, 'checkpoints')
TEST_PLOTS_DIR = os.path.join(TEST_DIR, 'plots')
TEST_RESULTS_DIR = os.path.join(TEST_DIR, 'results')
TEST_DATA_DIR = os.path.join(TEST_DIR, 'data')


def setup_test_dirs():
    """Create test directories."""
    for d in [TEST_DIR, TEST_CHECKPOINT_DIR, TEST_PLOTS_DIR, TEST_RESULTS_DIR, TEST_DATA_DIR]:
        os.makedirs(d, exist_ok=True)


def cleanup_test_dirs():
    """Remove test directories."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


# ============================================================================
# Test Utilities
# ============================================================================

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        msg = f"[{status}] {self.name} ({self.duration:.2f}s)"
        if self.error:
            msg += f"\n       Error: {self.error}"
        return msg


def run_test(test_func):
    """Run a test function and return TestResult."""
    result = TestResult(test_func.__name__)
    start = time.time()
    try:
        test_func()
        result.passed = True
    except Exception as e:
        result.error = str(e)
        if '--verbose' in sys.argv:
            traceback.print_exc()
    result.duration = time.time() - start
    return result


def create_synthetic_games(n_games=100, max_moves=80, output_path=None):
    """Create synthetic game data for testing.

    Returns:
        moves: (n_games, max_moves) array
        lengths: (n_games,) array
        winners: (n_games,) array
    """
    import domineering_game as dg

    moves_list = []
    lengths = []
    winners = []

    for i in range(n_games):
        np.random.seed(i)
        game = dg.domineering_game()
        game_moves = []

        while not game[3] and len(game_moves) < max_moves:
            legal = dg.legal_moves(game)
            legal_idx = np.where(legal)[0]
            if len(legal_idx) == 0:
                break
            move = np.random.choice(legal_idx)
            game_moves.append(move)
            dg.make_move(game, move)

        moves_list.append(game_moves)
        lengths.append(len(game_moves))
        winners.append(game[1] if game[3] else np.random.choice([True, False]))

    # Pad to max length
    max_len = max(lengths)
    moves_array = np.full((n_games, max_len), -1, dtype=np.int16)
    for i, m in enumerate(moves_list):
        moves_array[i, :len(m)] = m

    if output_path:
        np.savez_compressed(output_path,
                          moves=moves_array,
                          lengths=np.array(lengths, dtype=np.int16),
                          winners=np.array(winners, dtype=bool))

    return moves_array, np.array(lengths), np.array(winners)


# ============================================================================
# Unit Tests
# ============================================================================

def test_imports():
    """Test that all modules import correctly."""
    import config
    import model
    import data_loader
    import training
    import embedding
    import probing
    import run_experiment
    import selfplay


def test_config():
    """Test config values are correct."""
    from config import LR, SMALL_CONFIG, LARGE_CONFIG

    assert LR == 2e-4, f"LR should be 2e-4, got {LR}"
    assert SMALL_CONFIG['n_layers'] == 3, f"Small should have 3 layers"
    assert LARGE_CONFIG['n_layers'] == 6, f"Large should have 6 layers"
    assert SMALL_CONFIG['d_model'] == LARGE_CONFIG['d_model'], "d_model must match"
    assert SMALL_CONFIG['d_head'] == LARGE_CONFIG['d_head'], "d_head must match"


def test_model_creation():
    """Test model creation and parameter counts."""
    from model import create_small_model, create_large_model, count_parameters

    small = create_small_model()
    large = create_large_model()

    small_params = count_parameters(small)
    large_params = count_parameters(large)

    assert small_params > 0, "Small model should have parameters"
    assert large_params > small_params, "Large model should have more parameters"

    # Test forward pass
    tokens = torch.randint(0, 2, (2, 257))
    tokens[:, -1] = 2  # CLS token
    mask = torch.ones(2, 480, dtype=torch.bool)

    small_out = small(tokens, mask)
    large_out = large(tokens, mask)

    assert len(small_out) == 3, "Should return (value, policy, sector)"
    assert len(large_out) == 3, "Should return (value, policy, sector)"

    # Check shapes
    assert small_out[0].shape == (2, 1), f"Value shape wrong: {small_out[0].shape}"
    assert small_out[1].shape == (2, 480), f"Policy shape wrong: {small_out[1].shape}"
    assert small_out[2].shape == (2, 16), f"Sector shape wrong: {small_out[2].shape}"


def test_data_loader():
    """Test data loading and dataset creation."""
    from data_loader import DomineeringDataset, load_games, split_games

    # Create test data
    test_path = os.path.join(TEST_DATA_DIR, 'test_games.npz')
    create_synthetic_games(50, output_path=test_path)

    # Test loading
    moves, lengths, winners = load_games(test_path)
    assert len(moves) == 50, f"Expected 50 games, got {len(moves)}"

    # Test splitting
    splits = split_games(moves, lengths)
    assert 'train' in splits and 'val' in splits and 'test' in splits

    # Test dataset
    dataset = DomineeringDataset(test_path, split='train', positions_per_game=5)
    assert len(dataset) > 0, "Dataset should have items"

    # Test __getitem__
    item = dataset[0]
    assert 'tokens' in item, "Item should have tokens"
    assert 'value' in item, "Item should have value"
    assert 'policy' in item, "Item should have policy"
    assert 'mask' in item, "Item should have mask"
    assert 'sectors' in item, "Item should have sectors"

    assert item['tokens'].shape == (257,), f"Tokens shape wrong: {item['tokens'].shape}"
    assert item['mask'].shape == (480,), f"Mask shape wrong: {item['mask'].shape}"
    assert item['sectors'].shape == (16,), f"Sectors shape wrong: {item['sectors'].shape}"


def test_collate_batch():
    """Test batch collation."""
    from data_loader import DomineeringDataset
    from training import collate_batch
    from torch.utils.data import DataLoader

    test_path = os.path.join(TEST_DATA_DIR, 'test_games.npz')
    create_synthetic_games(50, output_path=test_path)

    dataset = DomineeringDataset(test_path, split='train', positions_per_game=5)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_batch)

    batch = next(iter(loader))
    assert 'tokens' in batch, "Batch should have tokens"
    assert batch['tokens'].shape[0] == 4, f"Batch size wrong: {batch['tokens'].shape[0]}"
    assert batch['tokens'].shape[1] == 257, f"Token dim wrong: {batch['tokens'].shape[1]}"


def test_compute_losses():
    """Test loss computation."""
    from training import compute_losses
    from model import create_small_model

    model = create_small_model()

    # Create fake batch
    batch = {
        'tokens': torch.randint(0, 2, (4, 257)),
        'mask': torch.ones(4, 480, dtype=torch.bool),
        'value': torch.rand(4),
        'policy': torch.randint(0, 480, (4,)),
        'sectors': torch.randn(4, 16),
    }
    batch['tokens'][:, -1] = 2

    value_pred, policy_logits, sector_pred = model(batch['tokens'], batch['mask'])

    # With auxiliary
    total, v, p, s = compute_losses(value_pred, policy_logits, sector_pred, batch, use_auxiliary=True)
    assert total.item() > 0, "Total loss should be positive"
    assert not torch.isnan(total), "Loss should not be NaN"

    # Without auxiliary
    total_noaux, v2, p2, s2 = compute_losses(value_pred, policy_logits, sector_pred, batch, use_auxiliary=False)
    assert total_noaux.item() > 0, "Total loss should be positive"


def test_quick_validate():
    """Test quick validation function."""
    from run_experiment import quick_validate
    from model import create_small_model
    from data_loader import DomineeringDataset
    from training import collate_batch
    from torch.utils.data import DataLoader

    test_path = os.path.join(TEST_DATA_DIR, 'test_games.npz')
    create_synthetic_games(50, output_path=test_path)

    model = create_small_model()
    dataset = DomineeringDataset(test_path, split='val', positions_per_game=5)
    loader = DataLoader(dataset, batch_size=10, collate_fn=collate_batch)

    loss = quick_validate(model, loader, device='cpu', use_auxiliary=False, n_batches=2)
    assert loss > 0, f"Loss should be positive, got {loss}"
    assert not np.isnan(loss), "Loss should not be NaN"


def test_embedding():
    """Test embedding small model into large model."""
    from model import create_small_model, create_large_model
    from embedding import embed_small_into_large, verify_embedding

    small = create_small_model()
    large = create_large_model()

    # Do embedding
    embed_small_into_large(small, large)

    # Verify
    tokens = torch.randint(0, 2, (4, 257))
    tokens[:, -1] = 2
    result = verify_embedding(small, large, tokens)

    assert result['value_diff'] < 1e-5, f"Value diff too large: {result['value_diff']}"
    assert result['policy_diff'] < 1e-5, f"Policy diff too large: {result['policy_diff']}"
    assert result['sector_diff'] < 1e-5, f"Sector diff too large: {result['sector_diff']}"


def test_probing():
    """Test probing mechanism."""
    from probing import train_probes_all_layers, LinearProbe
    from model import create_small_model
    from data_loader import DomineeringDataset
    from training import collate_batch
    from torch.utils.data import DataLoader

    test_path = os.path.join(TEST_DATA_DIR, 'test_games.npz')
    create_synthetic_games(100, output_path=test_path)

    model = create_small_model()
    dataset = DomineeringDataset(test_path, split='val', positions_per_game=10)
    loader = DataLoader(dataset, batch_size=20, collate_fn=collate_batch)

    probes = train_probes_all_layers(model, loader, n_samples=50, device='cpu')

    assert len(probes) > 0, "Should return at least one probe"
    for layer_idx, probe in probes.items():
        assert hasattr(probe, 'val_r2'), f"Probe should have val_r2"
        assert hasattr(probe, 'train_r2'), f"Probe should have train_r2"
        # R² can be negative for very bad fits, but should be reasonable
        assert -1 <= probe.val_r2 <= 1, f"val_r2 out of range: {probe.val_r2}"


def test_merge_npz():
    """Test merging NPZ files."""
    from run_experiment import merge_npz_files

    # Create two test files
    path1 = os.path.join(TEST_DATA_DIR, 'games1.npz')
    path2 = os.path.join(TEST_DATA_DIR, 'games2.npz')
    merged_path = os.path.join(TEST_DATA_DIR, 'merged.npz')

    create_synthetic_games(30, output_path=path1)
    create_synthetic_games(20, output_path=path2)

    merge_npz_files([path1, path2], merged_path)

    # Verify merged file
    data = np.load(merged_path)
    assert len(data['moves']) == 50, f"Expected 50 games, got {len(data['moves'])}"


def test_save_histories():
    """Test history saving and loading."""
    from run_experiment import save_histories

    # Create mock history
    histories = {
        'test_model': {
            'train_loss': [1.0, 0.9, 0.8],
            'val_loss': [1.1, 1.0, 0.9],
            'step_train_loss': [(100, 1.0), (200, 0.95)],
            'step_val_loss': [(500, 1.05)],
            'probe_steps': [1000],
            'probe_r2': {0: [0.5], 1: [0.6]},
            'model_name': 'test_model',
            'use_auxiliary': False,
        }
    }

    save_path = os.path.join(TEST_RESULTS_DIR, 'test_histories.json')
    save_histories(histories, save_path)

    assert os.path.exists(save_path), "Save file should exist"

    # Load and verify
    with open(save_path) as f:
        loaded = json.load(f)

    assert 'test_model' in loaded, "Should have test_model key"
    assert loaded['test_model']['train_loss'] == [1.0, 0.9, 0.8]


def test_incremental_plots():
    """Test incremental plot generation."""
    from run_experiment import update_plots_incrementally, update_probe_plot_incrementally

    # Create mock histories
    histories = {
        'Large-baseline_s42': {
            'step_train_loss': [(100, 1.0), (200, 0.9), (300, 0.85)],
            'step_val_loss': [(500, 1.1)],
            'probe_steps': [1000],
            'probe_r2': {0: [0.3], 1: [0.4], 2: [0.5]},
        },
        'Large+embed(aux)_s42': {
            'step_train_loss': [(100, 0.9), (200, 0.85), (300, 0.8)],
            'step_val_loss': [(500, 1.0)],
            'probe_steps': [1000],
            'probe_r2': {0: [0.4], 1: [0.5], 2: [0.6]},
        },
    }

    update_plots_incrementally(histories, TEST_PLOTS_DIR)
    update_probe_plot_incrementally(histories, TEST_PLOTS_DIR)

    loss_plot = os.path.join(TEST_PLOTS_DIR, 'loss_curves_live.png')
    probe_plot = os.path.join(TEST_PLOTS_DIR, 'probe_r2_live.png')

    assert os.path.exists(loss_plot), "Loss plot should exist"
    assert os.path.exists(probe_plot), "Probe plot should exist"
    assert os.path.getsize(loss_plot) > 0, "Loss plot should not be empty"
    assert os.path.getsize(probe_plot) > 0, "Probe plot should not be empty"


# ============================================================================
# Integration Test
# ============================================================================

def test_integration_mini_experiment():
    """Run a minimal end-to-end experiment.

    This tests the full pipeline with:
    - 100 synthetic games
    - 30 training steps per model
    - Intervals: train_log=5, quick_val=10, probe=15
    - 1 seed only
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST: Mini End-to-End Experiment")
    print("="*60)

    from model import create_small_model, create_large_model, count_parameters
    from data_loader import DomineeringDataset
    from training import collate_batch
    from embedding import embed_small_into_large, verify_embedding
    from run_experiment import (
        train_with_probes, quick_validate, save_histories,
        update_plots_incrementally, update_probe_plot_incrementally,
        plot_loss_curves, plot_probe_r2, plot_final_probe_comparison
    )
    from torch.utils.data import DataLoader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create test data
    print("\n[1/7] Creating test data...")
    test_data_path = os.path.join(TEST_DATA_DIR, 'integration_games.npz')
    create_synthetic_games(100, output_path=test_data_path)

    train_dataset = DomineeringDataset(test_data_path, split='train', positions_per_game=10)
    val_dataset = DomineeringDataset(test_data_path, split='val', positions_per_game=5)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, collate_fn=collate_batch)

    all_histories = {}
    seed = 42

    # Test parameters - smaller intervals to trigger all code paths
    n_steps = 30
    train_log_interval = 5
    quick_val_interval = 10
    probe_interval = 15

    # We need to modify train_with_probes to work with steps instead of epochs
    # For integration test, we'll call a simplified version

    def train_n_steps(model, model_name, use_auxiliary, n_steps):
        """Train for exactly n_steps."""
        from config import LR, WEIGHT_DECAY
        from training import compute_losses

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_value_loss': [],
            'val_policy_loss': [],
            'val_sector_loss': [],
            'steps': [],
            'step_train_loss': [],
            'step_val_loss': [],
            'probe_steps': [],
            'probe_r2': defaultdict(list),
            'model_name': model_name,
            'use_auxiliary': use_auxiliary,
        }

        model.train()
        step = 0
        recent_losses = []

        train_iter = iter(train_loader)

        while step < n_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()
            value_pred, policy_logits, sector_pred = model(batch['tokens'], batch['mask'])
            total_loss, v_loss, p_loss, s_loss = compute_losses(
                value_pred, policy_logits, sector_pred, batch, use_auxiliary=use_auxiliary)
            total_loss.backward()
            optimizer.step()

            step += 1
            recent_losses.append(total_loss.item())

            # Log training loss
            if step % train_log_interval == 0:
                avg_loss = sum(recent_losses[-train_log_interval:]) / min(len(recent_losses), train_log_interval)
                history['step_train_loss'].append((step, avg_loss))

            # Quick validation
            if step % quick_val_interval == 0:
                qloss = quick_validate(model, val_loader, device, use_auxiliary=use_auxiliary)
                history['step_val_loss'].append((step, qloss))
                model.train()

                # Update plots
                update_plots_incrementally(all_histories, TEST_PLOTS_DIR)

            # Probing
            if step % probe_interval == 0:
                from probing import train_probes_all_layers
                model.eval()
                probes = train_probes_all_layers(model, val_loader, n_samples=30, device=device)
                history['probe_steps'].append(step)
                for layer_idx, probe in probes.items():
                    history['probe_r2'][layer_idx].append(probe.val_r2)
                model.train()

                update_probe_plot_incrementally(all_histories, TEST_PLOTS_DIR)

        # Epoch-level metrics (just final values for test)
        history['train_loss'].append(sum(recent_losses) / len(recent_losses))
        history['steps'].append(step)

        # Final validation
        from training import evaluate
        model.eval()
        val_metrics = evaluate(model, val_loader, device, use_auxiliary=use_auxiliary)
        history['val_loss'].append(val_metrics['loss'])
        history['val_value_loss'].append(val_metrics['value_loss'])
        history['val_policy_loss'].append(val_metrics['policy_loss'])
        history['val_sector_loss'].append(val_metrics['sector_loss'])

        # Final probes
        from probing import train_probes_all_layers
        probes = train_probes_all_layers(model, val_loader, n_samples=50, device=device)
        history['final_probes'] = probes

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(TEST_CHECKPOINT_DIR, f'{model_name}_best.pt'))
        torch.save(model.state_dict(), os.path.join(TEST_CHECKPOINT_DIR, f'{model_name}_final.pt'))

        return history

    # Train small_aux
    print("\n[2/7] Training small_aux...")
    torch.manual_seed(seed)
    small_aux = create_small_model()
    hist = train_n_steps(small_aux, f'small_aux_s{seed}', use_auxiliary=True, n_steps=n_steps)
    all_histories[f'Small+aux_s{seed}'] = hist
    print(f"  Final val loss: {hist['val_loss'][-1]:.4f}")

    # Train small_noaux
    print("\n[3/7] Training small_noaux...")
    torch.manual_seed(seed)
    small_noaux = create_small_model()
    hist = train_n_steps(small_noaux, f'small_noaux_s{seed}', use_auxiliary=False, n_steps=n_steps)
    all_histories[f'Small-noaux_s{seed}'] = hist
    print(f"  Final val loss: {hist['val_loss'][-1]:.4f}")

    # Train large_baseline (NO aux)
    print("\n[4/7] Training large_baseline (no aux)...")
    torch.manual_seed(seed)
    large_baseline = create_large_model()
    hist = train_n_steps(large_baseline, f'large_baseline_s{seed}', use_auxiliary=False, n_steps=n_steps)
    all_histories[f'Large-baseline_s{seed}'] = hist
    print(f"  Final val loss: {hist['val_loss'][-1]:.4f}")

    # Embed small_aux -> large
    print("\n[5/7] Training large_embed_aux (no aux)...")
    torch.manual_seed(seed)
    large_embed_aux = create_large_model()

    # Reload small_aux
    small_aux_fresh = create_small_model()
    small_aux_fresh.load_state_dict(
        torch.load(os.path.join(TEST_CHECKPOINT_DIR, f'small_aux_s{seed}_best.pt'), weights_only=True)
    )

    embed_small_into_large(small_aux_fresh, large_embed_aux)

    # Verify embedding
    tokens = torch.randint(0, 2, (4, 257))
    tokens[:, -1] = 2
    verify_result = verify_embedding(small_aux_fresh, large_embed_aux, tokens)
    max_diff = max(verify_result['value_diff'], verify_result['policy_diff'], verify_result['sector_diff'])
    print(f"  Embedding verified: max diff = {max_diff:.2e}")
    assert max_diff < 1e-5, f"Embedding verification failed: max_diff = {max_diff}"

    hist = train_n_steps(large_embed_aux, f'large_embed_aux_s{seed}', use_auxiliary=False, n_steps=n_steps)
    all_histories[f'Large+embed(aux)_s{seed}'] = hist
    print(f"  Final val loss: {hist['val_loss'][-1]:.4f}")

    # Embed small_noaux -> large
    print("\n[6/7] Training large_embed_noaux (no aux)...")
    torch.manual_seed(seed)
    large_embed_noaux = create_large_model()

    small_noaux_fresh = create_small_model()
    small_noaux_fresh.load_state_dict(
        torch.load(os.path.join(TEST_CHECKPOINT_DIR, f'small_noaux_s{seed}_best.pt'), weights_only=True)
    )

    embed_small_into_large(small_noaux_fresh, large_embed_noaux)

    verify_result = verify_embedding(small_noaux_fresh, large_embed_noaux, tokens)
    max_diff = max(verify_result['value_diff'], verify_result['policy_diff'], verify_result['sector_diff'])
    print(f"  Embedding verified: max diff = {max_diff:.2e}")
    assert max_diff < 1e-5, f"Embedding verification failed: max_diff = {max_diff}"

    hist = train_n_steps(large_embed_noaux, f'large_embed_noaux_s{seed}', use_auxiliary=False, n_steps=n_steps)
    all_histories[f'Large+embed(noaux)_s{seed}'] = hist
    print(f"  Final val loss: {hist['val_loss'][-1]:.4f}")

    # Generate final plots and save histories
    print("\n[7/7] Generating final outputs...")
    save_histories(all_histories, os.path.join(TEST_RESULTS_DIR, 'histories.json'))
    plot_loss_curves(all_histories, os.path.join(TEST_PLOTS_DIR, 'loss_curves.png'))
    plot_probe_r2(all_histories, os.path.join(TEST_PLOTS_DIR, 'probe_r2.png'))
    plot_final_probe_comparison(all_histories, os.path.join(TEST_PLOTS_DIR, 'probe_final.png'))

    # Verify outputs
    assert os.path.exists(os.path.join(TEST_RESULTS_DIR, 'histories.json')), "Histories not saved"
    assert os.path.exists(os.path.join(TEST_PLOTS_DIR, 'loss_curves.png')), "Loss curves not saved"
    assert os.path.exists(os.path.join(TEST_PLOTS_DIR, 'probe_r2.png')), "Probe R² not saved"
    assert os.path.exists(os.path.join(TEST_PLOTS_DIR, 'probe_final.png')), "Probe final not saved"

    # Check all models trained
    assert len(all_histories) == 5, f"Expected 5 models, got {len(all_histories)}"

    # Check all histories have required keys
    required_keys = ['train_loss', 'val_loss', 'step_train_loss', 'step_val_loss',
                     'probe_steps', 'probe_r2', 'final_probes']
    for name, hist in all_histories.items():
        for key in required_keys:
            assert key in hist, f"Missing key '{key}' in {name}"

    print("\n" + "="*60)
    print("INTEGRATION TEST PASSED!")
    print("="*60)


def test_selfplay_generation():
    """Test self-play game generation."""
    from selfplay import generate_with_trained_model
    from model import create_small_model

    # Create and save a model
    model = create_small_model()
    model_path = os.path.join(TEST_CHECKPOINT_DIR, 'test_model.pt')
    torch.save(model.state_dict(), model_path)

    output_path = os.path.join(TEST_DATA_DIR, 'selfplay_test.npz')

    # Generate a few games
    stats = generate_with_trained_model(
        model_path=model_path,
        n_games=10,
        output_path=output_path,
        n_parallel=4,
        device='cpu',
        model_type='small'
    )

    assert os.path.exists(output_path), "Output file should exist"
    assert stats['n_games'] == 10, f"Expected 10 games, got {stats['n_games']}"

    # Verify NPZ format
    data = np.load(output_path)
    assert 'moves' in data, "Should have moves"
    assert 'lengths' in data, "Should have lengths"
    assert 'winners' in data, "Should have winners"


# ============================================================================
# Main
# ============================================================================

def run_unit_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("UNIT TESTS")
    print("="*60)

    setup_test_dirs()

    tests = [
        test_imports,
        test_config,
        test_model_creation,
        test_data_loader,
        test_collate_batch,
        test_compute_losses,
        test_quick_validate,
        test_embedding,
        test_probing,
        test_merge_npz,
        test_save_histories,
        test_incremental_plots,
    ]

    results = []
    for test in tests:
        result = run_test(test)
        results.append(result)
        print(result)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "-"*60)
    print(f"Unit Tests: {passed}/{total} passed")

    if passed < total:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")

    return passed == total


def run_integration_test():
    """Run integration test."""
    setup_test_dirs()
    result = run_test(test_integration_mini_experiment)
    print(result)
    return result.passed


def run_selfplay_test():
    """Run self-play test."""
    setup_test_dirs()
    result = run_test(test_selfplay_generation)
    print(result)
    return result.passed


def main():
    parser = argparse.ArgumentParser(description='Test experiment pipeline')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration test only')
    parser.add_argument('--selfplay', action='store_true', help='Run selfplay test only')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', action='store_true', help='Show full tracebacks')
    parser.add_argument('--keep', action='store_true', help='Keep test artifacts')

    args = parser.parse_args()

    # Default to all if nothing specified
    if not (args.unit or args.integration or args.selfplay or args.all):
        args.all = True

    all_passed = True

    try:
        if args.unit or args.all:
            if not run_unit_tests():
                all_passed = False

        if args.integration or args.all:
            if not run_integration_test():
                all_passed = False

        if args.selfplay or args.all:
            if not run_selfplay_test():
                all_passed = False

    finally:
        if not args.keep:
            print("\nCleaning up test artifacts...")
            cleanup_test_dirs()

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED - Ready for full experiment!")
    else:
        print("SOME TESTS FAILED - Fix issues before running experiment")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
