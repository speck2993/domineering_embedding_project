"""Main experiment runner for the embedding study.

Trains 5 models for seed=42:
1. Small+aux: Small model with auxiliary sector task
2. Small-noaux: Small model without auxiliary task
3. Large-baseline: Large model from scratch, with auxiliary
4. Large+embed(small+aux): Large with embedded small+aux, with auxiliary
5. Large+embed(small-noaux): Large with embedded small-noaux, with auxiliary

Collects probe R² throughout training and generates visualizations.
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from config import BATCH_SIZE, SMALL_CONFIG, LARGE_CONFIG
from model import create_small_model, create_large_model, count_parameters
from data_loader import DomineeringDataset
from training import train_model, collate_batch, evaluate
from embedding import embed_small_into_large, verify_embedding
from probing import train_probes_all_layers, print_probe_summary


# ============================================================================
# Data Utilities
# ============================================================================

def merge_npz_files(paths, output_path):
    """Merge multiple NPZ game files into one."""
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


# ============================================================================
# Training with Probing
# ============================================================================

def train_with_probes(model, train_loader, val_loader, n_epochs, model_name,
                      use_auxiliary, device, probe_interval=1000,
                      checkpoint_dir='checkpoints'):
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
        checkpoint_dir: Where to save checkpoints

    Returns:
        Dict with training history and probe results
    """
    from training import compute_losses
    from config import LR, WEIGHT_DECAY

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_value_loss': [],
        'val_policy_loss': [],
        'val_sector_loss': [],
        'steps': [],
        'probe_steps': [],
        'probe_r2': defaultdict(list),  # layer_idx -> list of R² values
    }

    global_step = 0
    best_val_loss = float('inf')

    print(f"\nTraining {model_name}...")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Epochs: {n_epochs}, Steps/epoch: {len(train_loader)}, Total: {total_steps}")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", ncols=100, mininterval=1.0)
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

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

            # Update progress bar (show total weighted loss + components)
            postfix = {'loss': f'{total_loss.item():.3f}', 'v': f'{v_loss.item():.3f}', 'p': f'{p_loss.item():.2f}'}
            if use_auxiliary:
                postfix['s'] = f'{s_loss.item():.2f}'
            pbar.set_postfix(postfix)

            # Probe evaluation
            if global_step % probe_interval == 0:
                model.eval()
                probes = train_probes_all_layers(model, val_loader, n_samples=1000,
                                                 device=device)
                history['probe_steps'].append(global_step)
                for layer_idx, probe in probes.items():
                    history['probe_r2'][layer_idx].append(probe.val_r2)
                model.train()

        pbar.close()

        # End of epoch
        train_loss = epoch_loss / epoch_samples
        history['train_loss'].append(train_loss)
        history['steps'].append(global_step)

        # Validation
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

    # Final probe evaluation
    model.eval()
    probes = train_probes_all_layers(model, val_loader, n_samples=2000, device=device)
    history['final_probes'] = probes

    # Save final checkpoint
    torch.save(model.state_dict(), f"{checkpoint_dir}/{model_name}_final.pt")

    return history


# ============================================================================
# Visualization
# ============================================================================

def plot_loss_curves(histories, output_path='plots/loss_curves.png'):
    """Plot training and validation loss curves for all models."""
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
# Main Experiment
# ============================================================================

def run_experiment(phase0_path='data/phase0_games.npz',
                   selfplay_path='data/selfplay_games.npz',
                   n_epochs=5, seed=42, device='cuda',
                   probe_interval=1000):
    """Run the full embedding experiment.

    Args:
        phase0_path: Path to phase0 games
        selfplay_path: Path to selfplay games
        n_epochs: Epochs per model
        seed: Random seed
        device: Training device
        probe_interval: Steps between probe evaluations

    Returns:
        Dict with all training histories
    """
    print("=" * 60)
    print("EMBEDDING EXPERIMENT")
    print("=" * 60)

    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -------------------------------------------------------------------------
    # Step 1: Prepare data
    # -------------------------------------------------------------------------
    print("\n[1/6] Preparing data...")

    combined_path = 'data/combined_for_experiment.npz'
    if not os.path.exists(combined_path):
        merge_npz_files([phase0_path, selfplay_path], combined_path)
    else:
        print(f"  Using existing {combined_path}")

    # Create data loaders
    # Use more positions per game for large models (~116 avg positions/game available)
    train_dataset = DomineeringDataset(combined_path, split='train', positions_per_game=50)
    val_dataset = DomineeringDataset(combined_path, split='val', positions_per_game=20)

    n_workers = 4 if device == 'cuda' else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=n_workers, collate_fn=collate_batch,
                              pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=n_workers, collate_fn=collate_batch,
                            pin_memory=(device == 'cuda'))

    print(f"  Train: {len(train_dataset)} positions, Val: {len(val_dataset)} positions")

    histories = {}

    # -------------------------------------------------------------------------
    # Step 2: Train Small+aux
    # -------------------------------------------------------------------------
    print("\n[2/6] Training Small+aux...")
    small_aux = create_small_model()
    histories['Small+aux'] = train_with_probes(
        small_aux, train_loader, val_loader, n_epochs,
        model_name='small_aux', use_auxiliary=True, device=device,
        probe_interval=probe_interval
    )

    # -------------------------------------------------------------------------
    # Step 3: Train Small-noaux
    # -------------------------------------------------------------------------
    print("\n[3/6] Training Small-noaux...")
    torch.manual_seed(seed)  # Reset seed for fair comparison
    small_noaux = create_small_model()
    histories['Small-noaux'] = train_with_probes(
        small_noaux, train_loader, val_loader, n_epochs,
        model_name='small_noaux', use_auxiliary=False, device=device,
        probe_interval=probe_interval
    )

    # -------------------------------------------------------------------------
    # Step 4: Train Large baseline
    # -------------------------------------------------------------------------
    print("\n[4/6] Training Large-baseline...")
    torch.manual_seed(seed)
    large_baseline = create_large_model()
    histories['Large-baseline'] = train_with_probes(
        large_baseline, train_loader, val_loader, n_epochs,
        model_name='large_baseline', use_auxiliary=True, device=device,
        probe_interval=probe_interval
    )

    # -------------------------------------------------------------------------
    # Step 5: Train Large+embed(small+aux)
    # -------------------------------------------------------------------------
    print("\n[5/6] Training Large+embed(small+aux)...")
    torch.manual_seed(seed)
    large_embed_aux = create_large_model()

    # Reload small+aux weights
    small_aux_fresh = create_small_model()
    small_aux_fresh.load_state_dict(torch.load('checkpoints/small_aux_best.pt', weights_only=True))

    # Embed
    embed_small_into_large(small_aux_fresh, large_embed_aux)

    # Verify embedding
    tokens = torch.randint(0, 2, (4, 257))
    tokens[:, -1] = 2
    verify_result = verify_embedding(small_aux_fresh, large_embed_aux, tokens)
    print(f"  Embedding verified: max diff = {max(verify_result['value_diff'], verify_result['policy_diff'], verify_result['sector_diff']):.2e}")

    histories['Large+embed(aux)'] = train_with_probes(
        large_embed_aux, train_loader, val_loader, n_epochs,
        model_name='large_embed_aux', use_auxiliary=True, device=device,
        probe_interval=probe_interval
    )

    # -------------------------------------------------------------------------
    # Step 6: Train Large+embed(small-noaux)
    # -------------------------------------------------------------------------
    print("\n[6/6] Training Large+embed(small-noaux)...")
    torch.manual_seed(seed)
    large_embed_noaux = create_large_model()

    # Reload small-noaux weights
    small_noaux_fresh = create_small_model()
    small_noaux_fresh.load_state_dict(torch.load('checkpoints/small_noaux_best.pt', weights_only=True))

    # Embed
    embed_small_into_large(small_noaux_fresh, large_embed_noaux)

    # Verify
    verify_result = verify_embedding(small_noaux_fresh, large_embed_noaux, tokens)
    print(f"  Embedding verified: max diff = {max(verify_result['value_diff'], verify_result['policy_diff'], verify_result['sector_diff']):.2e}")

    histories['Large+embed(noaux)'] = train_with_probes(
        large_embed_noaux, train_loader, val_loader, n_epochs,
        model_name='large_embed_noaux', use_auxiliary=True, device=device,
        probe_interval=probe_interval
    )

    # -------------------------------------------------------------------------
    # Generate visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    plot_loss_curves(histories)
    plot_probe_r2(histories)
    plot_final_probe_comparison(histories)

    # Print summary
    print("\nFinal Validation Loss:")
    print("-" * 40)
    for name, hist in histories.items():
        print(f"  {name:25}: {hist['val_loss'][-1]:.4f}")

    print("\nFinal Probe R² (layers 0, 1, final):")
    print("-" * 60)
    for name, hist in histories.items():
        if 'final_probes' in hist:
            probes = hist['final_probes']
            r2_0 = probes.get(0, type('', (), {'val_r2': 0})()).val_r2 if 0 in probes else 0
            r2_1 = probes.get(1, type('', (), {'val_r2': 0})()).val_r2 if 1 in probes else 0
            r2_f = probes.get(-1, type('', (), {'val_r2': 0})()).val_r2 if -1 in probes else 0
            print(f"  {name:25}: L0={r2_0:.4f}, L1={r2_1:.4f}, Final={r2_f:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    return histories


def main():
    parser = argparse.ArgumentParser(description='Run embedding experiment')
    parser.add_argument('--phase0', type=str, default='data/phase0_games.npz')
    parser.add_argument('--selfplay', type=str, default='data/selfplay_games.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--probe-interval', type=int, default=1000,
                        help='Steps between probe evaluations')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    run_experiment(
        phase0_path=args.phase0,
        selfplay_path=args.selfplay,
        n_epochs=args.epochs,
        seed=args.seed,
        device=device,
        probe_interval=args.probe_interval
    )


if __name__ == "__main__":
    main()
