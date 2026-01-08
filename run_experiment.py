"""Main experiment runner for the embedding study.

Trains models across multiple seeds:
- Small+aux: Small model with auxiliary sector task
- Small-noaux: Small model without auxiliary task
- Large-baseline: Large model from scratch, WITHOUT auxiliary
- Large+embed(small+aux): Large with embedded small+aux, WITHOUT auxiliary
- Large+embed(small-noaux): Large with embedded small-noaux, WITHOUT auxiliary

Large models are NOT trained with auxiliary task - we only probe them
to test whether sector representation persists from embedding.

Collects probe R² throughout training and generates visualizations
that are updated incrementally (can check during run).
"""

import argparse
import os
import time
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from config import BATCH_SIZE, SMALL_BATCH_SIZE, SMALL_CONFIG, LARGE_CONFIG
from model import create_small_model, create_large_model, count_parameters
from data_loader import EfficientDomineeringDataset
from training import train_model, collate_batch, evaluate, compute_losses
from embedding import embed_small_into_large, verify_embedding
from probing import train_probes_all_layers, print_probe_summary


# ============================================================================
# Color scheme for plots
# ============================================================================

MODEL_COLORS = {
    'Large-baseline': '#1f77b4',      # Blue
    'Large+embed(aux)': '#2ca02c',    # Green
    'Large+embed(noaux)': '#ff7f0e',  # Orange
    'Small+aux': '#9467bd',           # Purple
    'Small-noaux': '#8c564b',         # Brown
}


# ============================================================================
# Data Utilities
# ============================================================================

def merge_npz_files(paths, output_path):
    """Merge multiple NPZ game files into one."""
    all_moves = []
    all_lengths = []
    all_winners = []

    for path in paths:
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)")
            continue
        data = np.load(path)
        all_moves.append(data['moves'])
        all_lengths.append(data['lengths'])
        all_winners.append(data['winners'])
        print(f"  Loaded {path}: {len(data['moves'])} games")

    if not all_moves:
        raise ValueError("No game files found!")

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


# ============================================================================
# Quick Validation
# ============================================================================

def quick_validate(model, val_loader, device, use_auxiliary=False, n_batches=5, step=None):
    """Fast validation on a small subset (~1000 samples).

    Returns unbiased estimate of validation loss.
    Takes ~1-2 seconds.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device for evaluation
        use_auxiliary: Whether to include sector loss
        n_batches: Number of batches to evaluate (default 5 = ~1000 samples)
        step: If provided, seeds batch selection so all models at the same step
              evaluate on the same random batches (for fair comparison)

    Returns:
        float: Average loss over sampled batches
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    # Determine which batches to use
    total_batches = len(val_loader)
    if step is not None and total_batches > n_batches:
        # Seed by step so all models at same step get same batches
        rng = random.Random(step)
        batch_indices = set(rng.sample(range(total_batches), n_batches))
    else:
        # Just use first n_batches
        batch_indices = set(range(min(n_batches, total_batches)))

    batches_processed = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i not in batch_indices:
                continue
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            value_pred, policy_logits, sector_pred = model(batch['tokens'], batch['mask'])
            loss, _, _, _ = compute_losses(
                value_pred, policy_logits, sector_pred,
                batch, use_auxiliary=use_auxiliary)
            total_loss += loss.item() * len(batch['tokens'])
            n_samples += len(batch['tokens'])
            batches_processed += 1
            if batches_processed >= n_batches:
                break

    return total_loss / n_samples if n_samples > 0 else 0.0


# ============================================================================
# Training with Probing and Step Logging
# ============================================================================

def train_with_probes(model, train_loader, val_loader, n_epochs, model_name,
                      use_auxiliary, device, probe_interval=1000,
                      quick_val_interval=500, train_log_interval=100,
                      checkpoint_dir='checkpoints', all_histories=None,
                      plots_dir='plots'):
    """Train model and collect probe R² periodically.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        model_name: Name for logging/saving
        use_auxiliary: Whether to use auxiliary sector loss
        device: Training device
        probe_interval: Steps between probe evaluations
        quick_val_interval: Steps between quick validation checks
        train_log_interval: Steps between training loss logs
        checkpoint_dir: Where to save checkpoints
        all_histories: Dict of all histories (for incremental plot updates)
        plots_dir: Where to save plots

    Returns:
        Dict with training history and probe results
    """
    # Small models exist only to be embedded - skip expensive step-level tracking
    is_large_model = 'large' in model_name.lower()
    from config import LR, WEIGHT_DECAY

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # History tracking with step-level detail
    history = {
        'train_loss': [],               # Epoch-end train loss
        'val_loss': [],                 # Epoch-end val loss
        'val_value_loss': [],
        'val_policy_loss': [],
        'val_sector_loss': [],
        'steps': [],                    # Step count at epoch end
        'step_train_loss': [],          # (step, loss) tuples every train_log_interval
        'step_val_loss': [],            # (step, loss) tuples every quick_val_interval
        'probe_steps': [],
        'probe_r2': defaultdict(list),  # layer_idx -> list of R² values
        'model_name': model_name,
        'use_auxiliary': use_auxiliary,
    }

    global_step = 0
    best_val_loss = float('inf')
    recent_losses = []  # For smoothed train loss logging

    print(f"\nTraining {model_name}...")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Epochs: {n_epochs}, Steps/epoch: {len(train_loader)}, Total: {total_steps}")
    print(f"  Auxiliary task: {use_auxiliary}")
    if not is_large_model:
        print(f"  (Skipping step-level validation and probing for small model)")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", ncols=100, mininterval=1.0)
        for batch in pbar:
            # Move to device with non_blocking for better GPU utilization
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass
            optimizer.zero_grad()
            value_pred, policy_logits, sector_pred = model(batch['tokens'], batch['mask'])

            # Compute losses
            total_loss, v_loss, p_loss, s_loss = compute_losses(
                value_pred, policy_logits, sector_pred,
                batch, use_auxiliary=use_auxiliary)
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item() * len(batch['tokens'])
            epoch_samples += len(batch['tokens'])
            global_step += 1
            recent_losses.append(total_loss.item())

            # Update progress bar
            postfix = {'loss': f'{total_loss.item():.3f}', 'v': f'{v_loss.item():.3f}', 'p': f'{p_loss.item():.2f}'}
            if use_auxiliary:
                postfix['s'] = f'{s_loss.item():.2f}'
            pbar.set_postfix(postfix)

            # Log training loss every train_log_interval steps
            if global_step % train_log_interval == 0:
                avg_loss = sum(recent_losses[-train_log_interval:]) / len(recent_losses[-train_log_interval:])
                history['step_train_loss'].append((global_step, avg_loss))

            # Quick validation every quick_val_interval steps (only for large models)
            if is_large_model and global_step % quick_val_interval == 0:
                quick_loss = quick_validate(model, val_loader, device,
                                           use_auxiliary=use_auxiliary, step=global_step)
                history['step_val_loss'].append((global_step, quick_loss))
                model.train()

                # Update plots incrementally
                if all_histories is not None:
                    update_plots_incrementally(all_histories, plots_dir)

            # Probe evaluation every probe_interval steps (only for large models)
            if is_large_model and global_step % probe_interval == 0:
                model.eval()
                probes = train_probes_all_layers(model, val_loader, n_samples=1000,
                                                 device=device)
                history['probe_steps'].append(global_step)
                for layer_idx, probe in probes.items():
                    history['probe_r2'][layer_idx].append(probe.val_r2)
                model.train()

                # Update probe plots
                if all_histories is not None:
                    update_probe_plot_incrementally(all_histories, plots_dir)

        pbar.close()

        # End of epoch
        train_loss = epoch_loss / epoch_samples
        history['train_loss'].append(train_loss)
        history['steps'].append(global_step)

        # Full validation at epoch end
        model.eval()
        val_metrics = evaluate(model, val_loader, device, use_auxiliary=use_auxiliary)
        history['val_loss'].append(val_metrics['loss'])
        history['val_value_loss'].append(val_metrics['value_loss'])
        history['val_policy_loss'].append(val_metrics['policy_loss'])
        history['val_sector_loss'].append(val_metrics['sector_loss'])

        print(f"  Val: loss={val_metrics['loss']:.4f}, v_acc={val_metrics['value_acc']:.1%}, p_acc={val_metrics['policy_acc']:.1%}")

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), f"{checkpoint_dir}/{model_name}_best.pt")

        # Signal epoch end to dataset (for EfficientDomineeringDataset to refresh positions)
        if hasattr(train_loader.dataset, 'on_epoch_end'):
            train_loader.dataset.on_epoch_end()
            train_loader.dataset.precompute_epoch()

    # Final probe evaluation (only for large models)
    if is_large_model:
        model.eval()
        probes = train_probes_all_layers(model, val_loader, n_samples=2000, device=device)
        history['final_probes'] = probes

    # Save final checkpoint
    torch.save(model.state_dict(), f"{checkpoint_dir}/{model_name}_final.pt")

    return history


# ============================================================================
# Incremental Visualization
# ============================================================================

def update_plots_incrementally(all_histories, plots_dir='plots'):
    """Update loss curve plots as training progresses.

    This function can be called during training to update plots
    that you can check by opening the images.

    Args:
        all_histories: Dict mapping model_name -> history dict
        plots_dir: Directory to save plots
    """
    os.makedirs(plots_dir, exist_ok=True)

    # Group histories by model type for averaging
    type_histories = {
        'Large-baseline': [],
        'Large+embed(aux)': [],
        'Large+embed(noaux)': [],
    }

    for name, hist in all_histories.items():
        for type_name in type_histories.keys():
            if type_name in name:
                type_histories[type_name].append(hist)
                break

    # Create figure with two subplots: train loss and val loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training loss (left subplot)
    ax = axes[0]
    for type_name, hists in type_histories.items():
        if not hists:
            continue
        color = MODEL_COLORS.get(type_name, 'gray')

        # Plot individual models (translucent, no labels)
        for hist in hists:
            if 'step_train_loss' in hist and hist['step_train_loss']:
                steps, losses = zip(*hist['step_train_loss'])
                ax.plot(steps, losses, color=color, alpha=0.25, linewidth=1)

        # Plot running average (opaque, with label) if we have data
        all_data = []
        for hist in hists:
            if 'step_train_loss' in hist and hist['step_train_loss']:
                all_data.append(dict(hist['step_train_loss']))
        if all_data:
            all_steps = sorted(set().union(*[d.keys() for d in all_data]))
            avg_losses = []
            for step in all_steps:
                vals = [d[step] for d in all_data if step in d]
                avg_losses.append(np.mean(vals) if vals else np.nan)
            ax.plot(all_steps, avg_losses, color=color, alpha=1.0, linewidth=2, label=type_name)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    if ax.get_legend_handles_labels()[0]:  # Only add legend if there are labeled artists
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot validation loss (right subplot)
    ax = axes[1]
    for type_name, hists in type_histories.items():
        if not hists:
            continue
        color = MODEL_COLORS.get(type_name, 'gray')

        # Plot individual models (translucent, no labels)
        for hist in hists:
            if 'step_val_loss' in hist and hist['step_val_loss']:
                steps, losses = zip(*hist['step_val_loss'])
                ax.plot(steps, losses, color=color, alpha=0.25, linewidth=1)

        # Plot running average (opaque, with label) if we have data
        all_data = []
        for hist in hists:
            if 'step_val_loss' in hist and hist['step_val_loss']:
                all_data.append(dict(hist['step_val_loss']))
        if all_data:
            all_steps = sorted(set().union(*[d.keys() for d in all_data]))
            avg_losses = []
            for step in all_steps:
                vals = [d[step] for d in all_data if step in d]
                avg_losses.append(np.mean(vals) if vals else np.nan)
            ax.plot(all_steps, avg_losses, color=color, alpha=1.0, linewidth=2, label=type_name)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss (Quick Samples)')
    if ax.get_legend_handles_labels()[0]:  # Only add legend if there are labeled artists
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_curves_live.png'), dpi=150)
    plt.close()


def update_probe_plot_incrementally(all_histories, plots_dir='plots'):
    """Update probe R² plot as training progresses.

    Args:
        all_histories: Dict mapping model_name -> history dict
        plots_dir: Directory to save plots
    """
    os.makedirs(plots_dir, exist_ok=True)

    # Focus on layer 2 (the last embedded layer) for the main comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    type_histories = {
        'Large-baseline': [],
        'Large+embed(aux)': [],
        'Large+embed(noaux)': [],
    }

    for name, hist in all_histories.items():
        for type_name in type_histories.keys():
            if type_name in name:
                type_histories[type_name].append(hist)
                break

    target_layer = 2  # The last embedded layer

    for type_name, hists in type_histories.items():
        if not hists:
            continue
        color = MODEL_COLORS.get(type_name, 'gray')

        # Plot individual models (translucent, no labels)
        for hist in hists:
            probe_steps = hist.get('probe_steps', [])
            probe_r2 = hist.get('probe_r2', {})
            if target_layer in probe_r2 and probe_steps:
                ax.plot(probe_steps, probe_r2[target_layer],
                       color=color, alpha=0.25, linewidth=1, marker='o', markersize=2)

        # Plot running average (opaque, with label) if we have data
        all_data = []
        for hist in hists:
            probe_steps = hist.get('probe_steps', [])
            probe_r2 = hist.get('probe_r2', {})
            if target_layer in probe_r2 and probe_steps:
                all_data.append(dict(zip(probe_steps, probe_r2[target_layer])))
        if all_data:
            all_steps = sorted(set().union(*[d.keys() for d in all_data]))
            avg_r2 = []
            for step in all_steps:
                vals = [d[step] for d in all_data if step in d]
                avg_r2.append(np.mean(vals) if vals else np.nan)
            ax.plot(all_steps, avg_r2, color=color, alpha=1.0, linewidth=2,
                   marker='o', markersize=4, label=type_name)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Probe R²')
    ax.set_title(f'Probe R² at Layer {target_layer} (Last Embedded Layer)')
    if ax.get_legend_handles_labels()[0]:  # Only add legend if there are labeled artists
        ax.legend()
    ax.grid(True, alpha=0.3)
    # Auto-scale y-axis but ensure 0 and 1 are visible for reference
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(min(y_min, -0.1), max(y_max, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'probe_r2_live.png'), dpi=150)
    plt.close()


def plot_loss_curves(histories, output_path='plots/loss_curves.png'):
    """Plot final training and validation loss curves for all models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax = axes[0]
    for name, hist in histories.items():
        epochs = range(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[1]
    for name, hist in histories.items():
        epochs = range(1, len(hist['val_loss']) + 1)
        ax.plot(epochs, hist['val_loss'], label=name, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved loss curves to {output_path}")


def plot_probe_r2(histories, output_path='plots/probe_r2.png'):
    """Plot probe R² evolution for large models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Only plot large models (they have more layers)
    large_models = {k: v for k, v in histories.items() if 'Large' in k}

    if not large_models:
        print("No large models to plot probe R²")
        return

    n_models = len(large_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, hist) in zip(axes, large_models.items()):
        probe_steps = hist['probe_steps']
        probe_r2 = hist['probe_r2']

        for layer_idx in sorted(probe_r2.keys()):
            if layer_idx == -1:
                label = 'Final'
            else:
                label = f'Layer {layer_idx}'
            ax.plot(probe_steps, probe_r2[layer_idx], label=label, marker='o', markersize=2)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Probe R²')
        ax.set_title(f'{name} Probe R²')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved probe R² to {output_path}")


def plot_final_probe_comparison(histories, output_path='plots/probe_final.png'):
    """Compare final probe R² across models as bar chart."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect final probe R² for each model
    data = {}
    for name, hist in histories.items():
        if 'final_probes' in hist:
            probes = hist['final_probes']
            data[name] = {k: p.val_r2 for k, p in probes.items()}

    if not data:
        print("No final probe data to plot")
        return

    # Get all layer indices
    all_layers = sorted(set().union(*[set(d.keys()) for d in data.values()]))

    # Create bar chart
    x = np.arange(len(all_layers))
    width = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, layer_r2) in enumerate(data.items()):
        values = [layer_r2.get(l, 0) for l in all_layers]
        offset = (i - len(data)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Probe R²')
    ax.set_title('Final Probe R² by Layer')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' if l >= 0 else 'Final' for l in all_layers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved final probe comparison to {output_path}")


# ============================================================================
# Main Experiment - Single Seed
# ============================================================================

def run_single_seed(seed, train_loader, val_loader, n_epochs, device,
                    probe_interval, all_histories, checkpoint_dir='checkpoints',
                    plots_dir='plots', train_loader_small=None, val_loader_small=None):
    """Run experiment for a single seed.

    Args:
        seed: Random seed
        train_loader: Training data loader (for large models)
        val_loader: Validation data loader (for large models)
        n_epochs: Epochs per model
        device: Training device
        probe_interval: Steps between probe evaluations
        all_histories: Dict to accumulate all histories
        checkpoint_dir: Where to save checkpoints
        plots_dir: Where to save plots
        train_loader_small: Training data loader for small models (larger batch)
        val_loader_small: Validation data loader for small models (larger batch)

    Returns:
        Dict with histories for this seed
    """
    # Use provided small loaders or fall back to regular loaders
    if train_loader_small is None:
        train_loader_small = train_loader
    if val_loader_small is None:
        val_loader_small = val_loader
    seed_histories = {}

    # Training order designed for quick time estimation:
    # 1. Large baseline first (gives immediate sense of large model training time)
    # 2. Small noaux -> Large embed noaux (noaux pair)
    # 3. Small aux -> Large embed aux (aux pair)

    # -------------------------------------------------------------------------
    # 1. Train Large-baseline (NO auxiliary task)
    # -------------------------------------------------------------------------
    print(f"\n[Seed {seed}] Training Large-baseline (no aux)...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    large_baseline = create_large_model()
    model_name = f'large_baseline_s{seed}'
    hist = train_with_probes(
        large_baseline, train_loader, val_loader, n_epochs,
        model_name=model_name, use_auxiliary=False, device=device,
        probe_interval=probe_interval, checkpoint_dir=checkpoint_dir,
        all_histories=all_histories, plots_dir=plots_dir
    )
    seed_histories['Large-baseline'] = hist
    all_histories[f'Large-baseline_s{seed}'] = hist

    # -------------------------------------------------------------------------
    # 2a. Train Small-noaux
    # -------------------------------------------------------------------------
    print(f"\n[Seed {seed}] Training Small-noaux...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    small_noaux = create_small_model()
    model_name = f'small_noaux_s{seed}'
    hist = train_with_probes(
        small_noaux, train_loader_small, val_loader_small, n_epochs,
        model_name=model_name, use_auxiliary=False, device=device,
        probe_interval=probe_interval, checkpoint_dir=checkpoint_dir,
        all_histories=all_histories, plots_dir=plots_dir
    )
    seed_histories['Small-noaux'] = hist
    all_histories[f'Small-noaux_s{seed}'] = hist

    # -------------------------------------------------------------------------
    # 2b. Train Large+embed(small-noaux) (NO auxiliary task)
    # -------------------------------------------------------------------------
    print(f"\n[Seed {seed}] Training Large+embed(noaux) (no aux training)...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    large_embed_noaux = create_large_model()

    # Reload small-noaux weights
    small_noaux_fresh = create_small_model()
    small_noaux_fresh.load_state_dict(
        torch.load(f'{checkpoint_dir}/small_noaux_s{seed}_best.pt', weights_only=True)
    )

    # Embed
    embed_small_into_large(small_noaux_fresh, large_embed_noaux)

    # Verify embedding
    tokens = torch.randint(0, 2, (4, 257))
    tokens[:, -1] = 2
    verify_result = verify_embedding(small_noaux_fresh, large_embed_noaux, tokens)
    max_diff = max(verify_result['value_diff'], verify_result['policy_diff'], verify_result['sector_diff'])
    print(f"  Embedding verified: max diff = {max_diff:.2e}")

    model_name = f'large_embed_noaux_s{seed}'
    hist = train_with_probes(
        large_embed_noaux, train_loader, val_loader, n_epochs,
        model_name=model_name, use_auxiliary=False, device=device,
        probe_interval=probe_interval, checkpoint_dir=checkpoint_dir,
        all_histories=all_histories, plots_dir=plots_dir
    )
    seed_histories['Large+embed(noaux)'] = hist
    all_histories[f'Large+embed(noaux)_s{seed}'] = hist

    # -------------------------------------------------------------------------
    # 3a. Train Small+aux
    # -------------------------------------------------------------------------
    print(f"\n[Seed {seed}] Training Small+aux...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    small_aux = create_small_model()
    model_name = f'small_aux_s{seed}'
    hist = train_with_probes(
        small_aux, train_loader_small, val_loader_small, n_epochs,
        model_name=model_name, use_auxiliary=True, device=device,
        probe_interval=probe_interval, checkpoint_dir=checkpoint_dir,
        all_histories=all_histories, plots_dir=plots_dir
    )
    seed_histories['Small+aux'] = hist
    all_histories[f'Small+aux_s{seed}'] = hist

    # -------------------------------------------------------------------------
    # 3b. Train Large+embed(small+aux) (NO auxiliary task)
    # -------------------------------------------------------------------------
    print(f"\n[Seed {seed}] Training Large+embed(aux) (no aux training)...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    large_embed_aux = create_large_model()

    # Reload small+aux weights
    small_aux_fresh = create_small_model()
    small_aux_fresh.load_state_dict(
        torch.load(f'{checkpoint_dir}/small_aux_s{seed}_best.pt', weights_only=True)
    )

    # Embed
    embed_small_into_large(small_aux_fresh, large_embed_aux)

    # Verify
    verify_result = verify_embedding(small_aux_fresh, large_embed_aux, tokens)
    max_diff = max(verify_result['value_diff'], verify_result['policy_diff'], verify_result['sector_diff'])
    print(f"  Embedding verified: max diff = {max_diff:.2e}")

    model_name = f'large_embed_aux_s{seed}'
    hist = train_with_probes(
        large_embed_aux, train_loader, val_loader, n_epochs,
        model_name=model_name, use_auxiliary=False, device=device,
        probe_interval=probe_interval, checkpoint_dir=checkpoint_dir,
        all_histories=all_histories, plots_dir=plots_dir
    )
    seed_histories['Large+embed(aux)'] = hist
    all_histories[f'Large+embed(aux)_s{seed}'] = hist

    return seed_histories


# ============================================================================
# Full Experiment - Multiple Seeds
# ============================================================================

def run_full_experiment(phase0_path='data/phase0_games.npz',
                        selfplay_path='data/selfplay_games.npz',
                        additional_selfplay_path='data/selfplay_additional.npz',
                        n_epochs=3, seeds=[42, 123, 456], device='cuda',
                        probe_interval=1000):
    """Run the full embedding experiment across multiple seeds.

    Args:
        phase0_path: Path to phase0 games
        selfplay_path: Path to selfplay games
        additional_selfplay_path: Path to additional selfplay games (optional)
        n_epochs: Epochs per model
        seeds: List of random seeds
        device: Training device
        probe_interval: Steps between probe evaluations

    Returns:
        Dict with all training histories
    """
    print("=" * 60)
    print("EMBEDDING EXPERIMENT")
    print("=" * 60)
    print(f"Seeds: {seeds}")
    print(f"Epochs: {n_epochs}")
    print(f"Probe interval: {probe_interval}")

    start_time = time.time()

    # -------------------------------------------------------------------------
    # Step 1: Prepare data
    # -------------------------------------------------------------------------
    print("\n[1/2] Preparing data...")

    combined_path = 'data/combined_for_experiment.npz'
    data_files = [phase0_path, selfplay_path]
    if os.path.exists(additional_selfplay_path):
        data_files.append(additional_selfplay_path)

    # Always regenerate to pick up any new data
    merge_npz_files(data_files, combined_path)

    # Create data loaders
    # Use EfficientDomineeringDataset for both (precomputes positions)
    train_dataset = EfficientDomineeringDataset(combined_path, split='train', positions_per_game=60)
    val_dataset = EfficientDomineeringDataset(combined_path, split='val', positions_per_game=20)

    # Precompute positions now (before creating DataLoader)
    print("  Pre-computing training positions...")
    train_dataset.precompute_epoch()
    print("  Pre-computing validation positions...")
    val_dataset.precompute_epoch()

    # DataLoader settings:
    # - num_workers=0 since EfficientDataset does minimal work per __getitem__
    #   and we want to avoid duplicating precomputed data across worker processes
    # - Separate loaders for small models (larger batch) and large models
    pin = (device == 'cuda')

    # Large model loaders (standard batch size)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=collate_batch, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_batch, pin_memory=pin)

    # Small model loaders (larger batch for faster training)
    train_loader_small = DataLoader(train_dataset, batch_size=SMALL_BATCH_SIZE, shuffle=False,
                                    num_workers=0, collate_fn=collate_batch, pin_memory=pin)
    val_loader_small = DataLoader(val_dataset, batch_size=SMALL_BATCH_SIZE, shuffle=False,
                                  num_workers=0, collate_fn=collate_batch, pin_memory=pin)

    print(f"  Batch sizes: small={SMALL_BATCH_SIZE}, large={BATCH_SIZE}")

    print(f"  Train: {len(train_dataset)} positions, Val: {len(val_dataset)} positions")

    # -------------------------------------------------------------------------
    # Step 2: Train models across seeds
    # -------------------------------------------------------------------------
    print("\n[2/2] Training models...")

    all_histories = {}

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({i+1}/{len(seeds)})")
        print("=" * 60)

        run_single_seed(
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            train_loader_small=train_loader_small,
            val_loader_small=val_loader_small,
            n_epochs=n_epochs,
            device=device,
            probe_interval=probe_interval,
            all_histories=all_histories,
            checkpoint_dir='checkpoints',
            plots_dir='plots'
        )

        # Save intermediate results
        save_histories(all_histories, 'results/histories.json')

    # -------------------------------------------------------------------------
    # Generate final visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    plot_loss_curves(all_histories)
    plot_probe_r2(all_histories)
    plot_final_probe_comparison(all_histories)
    update_plots_incrementally(all_histories)
    update_probe_plot_incrementally(all_histories)

    # Print summary
    print("\nFinal Validation Loss:")
    print("-" * 40)
    for name, hist in sorted(all_histories.items()):
        if hist['val_loss']:
            print(f"  {name:30}: {hist['val_loss'][-1]:.4f}")

    print("\nFinal Probe R² at Layer 2 (embedded layer):")
    print("-" * 60)
    for name, hist in sorted(all_histories.items()):
        if 'final_probes' in hist and 2 in hist['final_probes']:
            r2 = hist['final_probes'][2].val_r2
            print(f"  {name:30}: {r2:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")

    return all_histories


def save_histories(histories, path):
    """Save histories to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert to JSON-serializable format
    serializable = {}
    for name, hist in histories.items():
        s_hist = {}
        for k, v in hist.items():
            if k == 'final_probes':
                s_hist[k] = {str(layer): {'val_r2': p.val_r2, 'train_r2': p.train_r2}
                            for layer, p in v.items()}
            elif k == 'probe_r2':
                s_hist[k] = {str(layer): list(r2s) for layer, r2s in v.items()}
            elif isinstance(v, defaultdict):
                s_hist[k] = dict(v)
            else:
                s_hist[k] = v
        serializable[name] = s_hist

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


# ============================================================================
# Legacy single-seed experiment (for compatibility)
# ============================================================================

def run_experiment(phase0_path='data/phase0_games.npz',
                   selfplay_path='data/selfplay_games.npz',
                   n_epochs=3, seed=42, device='cuda',
                   probe_interval=1000):
    """Run single-seed experiment (legacy compatibility)."""
    return run_full_experiment(
        phase0_path=phase0_path,
        selfplay_path=selfplay_path,
        n_epochs=n_epochs,
        seeds=[seed],
        device=device,
        probe_interval=probe_interval
    )


def main():
    parser = argparse.ArgumentParser(description='Run embedding experiment')
    parser.add_argument('--phase0', type=str, default='data/phase0_games.npz')
    parser.add_argument('--selfplay', type=str, default='data/selfplay_games.npz')
    parser.add_argument('--additional-selfplay', type=str, default='data/selfplay_additional.npz')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Comma-separated list of seeds')
    parser.add_argument('--probe-interval', type=int, default=1000,
                        help='Steps between probe evaluations')

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    run_full_experiment(
        phase0_path=args.phase0,
        selfplay_path=args.selfplay,
        additional_selfplay_path=args.additional_selfplay,
        n_epochs=args.epochs,
        seeds=seeds,
        device=device,
        probe_interval=args.probe_interval
    )


if __name__ == "__main__":
    main()
