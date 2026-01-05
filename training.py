"""Training loop for Domineering neural networks.

Implements Section 6 of the specification with:
- AdamW optimizer, cosine LR schedule, FP16 mixed precision
- Tensorboard logging
- Support for auxiliary task toggle
"""

import argparse
import os
import time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import LR, WEIGHT_DECAY, BATCH_SIZE, POLICY_LOSS_WEIGHT, SECTOR_LOSS_WEIGHT
from model import create_small_model, create_medium_model, create_large_model, count_parameters
from data_loader import DomineeringDataset


# ============================================================================
# Collate Function
# ============================================================================

def collate_batch(samples):
    """Stack samples into batched tensors.

    Args:
        samples: List of dicts from DomineeringDataset

    Returns:
        Dict with batched tensors on CPU (move to device in training loop)
    """
    return {
        'tokens': torch.stack([torch.from_numpy(s['tokens']) for s in samples]),
        'value': torch.tensor([s['value'] for s in samples], dtype=torch.float32),
        'policy': torch.tensor([s['policy'] for s in samples], dtype=torch.long),
        'mask': torch.stack([torch.from_numpy(s['mask']) for s in samples]),
        'sectors': torch.stack([torch.from_numpy(s['sectors']) for s in samples])
    }


# ============================================================================
# Loss Functions
# ============================================================================

def compute_losses(value_pred, policy_logits, sector_pred, batch,
                   use_auxiliary=True, value_only=False):
    """Compute training losses.

    Args:
        value_pred: (batch, 1) - model value predictions
        policy_logits: (batch, 480) - model policy logits (after masking)
        sector_pred: (batch, 16) - model sector predictions
        batch: Dict with 'value', 'policy', 'sectors' targets
        use_auxiliary: Whether to include sector loss
        value_only: If True, only compute value loss (for bootstrap self-play model)

    Returns:
        total_loss, value_loss, policy_loss, sector_loss (all scalar tensors)
    """
    # Value loss: BCE with logits (numerically stable for FP16)
    value_loss = F.binary_cross_entropy_with_logits(value_pred.squeeze(-1), batch['value'])

    # Value-only mode for bootstrap self-play model
    if value_only:
        zero = torch.tensor(0.0, device=value_loss.device)
        return value_loss, value_loss, zero, zero

    # Policy loss: Cross-entropy on the move actually played
    policy_loss = F.cross_entropy(policy_logits, batch['policy'])

    # Sector loss: MSE on sector balance predictions
    sector_loss = F.mse_loss(sector_pred, batch['sectors'])

    # Total loss (policy weighted to balance with value loss)
    if use_auxiliary:
        total_loss = value_loss + POLICY_LOSS_WEIGHT * policy_loss + SECTOR_LOSS_WEIGHT * sector_loss
    else:
        total_loss = value_loss + POLICY_LOSS_WEIGHT * policy_loss

    return total_loss, value_loss, policy_loss, sector_loss


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model, val_loader, device, use_auxiliary=True, value_only=False):
    """Evaluate model on validation set.

    Returns:
        Dict with losses and metrics
    """
    model.eval()
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    amp_context = autocast(device_type='cuda') if device_type == 'cuda' else nullcontext()

    total_loss = 0.0
    total_value_loss = 0.0
    total_policy_loss = 0.0
    total_sector_loss = 0.0
    total_value_correct = 0
    total_policy_correct = 0
    total_samples = 0

    for batch in val_loader:
        # Move to device
        tokens = batch['tokens'].to(device)
        mask = batch['mask'].to(device)
        value_target = batch['value'].to(device)
        policy_target = batch['policy'].to(device)
        sectors_target = batch['sectors'].to(device)

        # Forward pass
        with amp_context:
            value_pred, policy_logits, sector_pred = model(tokens, mask)
            loss, v_loss, p_loss, s_loss = compute_losses(
                value_pred, policy_logits, sector_pred,
                {'value': value_target, 'policy': policy_target, 'sectors': sectors_target},
                use_auxiliary, value_only
            )

        batch_size = tokens.shape[0]
        total_loss += loss.item() * batch_size
        total_value_loss += v_loss.item() * batch_size
        total_policy_loss += p_loss.item() * batch_size
        total_sector_loss += s_loss.item() * batch_size

        # Value accuracy (threshold at 0.5, apply sigmoid since model outputs logits)
        value_preds_binary = (torch.sigmoid(value_pred.squeeze(-1)) > 0.5).float()
        total_value_correct += (value_preds_binary == value_target).sum().item()

        # Policy top-1 accuracy
        policy_preds = policy_logits.argmax(dim=-1)
        total_policy_correct += (policy_preds == policy_target).sum().item()

        total_samples += batch_size

    return {
        'loss': total_loss / total_samples,
        'value_loss': total_value_loss / total_samples,
        'policy_loss': total_policy_loss / total_samples,
        'sector_loss': total_sector_loss / total_samples,
        'value_acc': total_value_correct / total_samples,
        'policy_acc': total_policy_correct / total_samples,
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_for_steps(model, train_loader, val_loader, n_steps,
                    use_auxiliary=True, value_only=False, device='cuda',
                    output_path=None, logdir=None, seed=42, silent=False):
    """Train a model for a fixed number of gradient steps.

    Cycles through train_loader as needed to reach n_steps.

    Args:
        model: DomineeringTransformer to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_steps: Number of gradient steps to train
        use_auxiliary: Whether to use auxiliary sector loss
        value_only: If True, only train value head
        device: Device to train on
        output_path: Path to save final checkpoint
        logdir: Directory for Tensorboard logs
        seed: Random seed
        silent: If True, suppress progress output

    Returns:
        Dict with final metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    amp_context = autocast(device_type='cuda') if device_type == 'cuda' else nullcontext()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)

    use_scaler = device_type == 'cuda'
    scaler = GradScaler() if use_scaler else None

    writer = None
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)

    if not silent:
        print(f"Training for {n_steps} steps")
        print(f"Auxiliary task: {use_auxiliary}, Value only: {value_only}")
        print(f"Parameters: {count_parameters(model):,}")

    model.train()
    train_iter = iter(train_loader)
    global_step = 0

    pbar = tqdm(total=n_steps, desc="Training", disable=silent, ncols=80, mininterval=1.0)
    while global_step < n_steps:
        # Get next batch, cycling if needed
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move to device
        tokens = batch['tokens'].to(device)
        mask = batch['mask'].to(device)
        value_target = batch['value'].to(device)
        policy_target = batch['policy'].to(device)
        sectors_target = batch['sectors'].to(device)

        # Forward pass
        optimizer.zero_grad()
        with amp_context:
            value_pred, policy_logits, sector_pred = model(tokens, mask)
            loss, v_loss, p_loss, s_loss = compute_losses(
                value_pred, policy_logits, sector_pred,
                {'value': value_target, 'policy': policy_target, 'sectors': sectors_target},
                use_auxiliary, value_only
            )

        # Backward pass
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Log to tensorboard
        if writer and global_step % 100 == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/value_loss', v_loss.item(), global_step)
            writer.add_scalar('train/policy_loss', p_loss.item(), global_step)
            writer.add_scalar('train/sector_loss', s_loss.item(), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        pbar.update(1)
        global_step += 1

    pbar.close()

    # Final validation
    val_metrics = evaluate(model, val_loader, device, use_auxiliary, value_only)

    if not silent:
        print(f"Final validation: loss={val_metrics['loss']:.4f} "
              f"v_acc={val_metrics['value_acc']:.3f} p_acc={val_metrics['policy_acc']:.3f}")

    # Save checkpoint
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'steps': n_steps,
            'val_loss': val_metrics['loss'],
            'use_auxiliary': use_auxiliary,
            'value_only': value_only,
        }, output_path)
        if not silent:
            print(f"Saved checkpoint to {output_path}")

    if writer:
        writer.close()

    return val_metrics


def train_model(model, train_loader, val_loader, n_epochs,
                use_auxiliary=True, value_only=False, device='cuda',
                output_path=None, logdir=None, seed=42):
    """Train a model.

    Args:
        model: DomineeringTransformer to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: Number of epochs to train
        use_auxiliary: Whether to use auxiliary sector loss
        value_only: If True, only train value head (for bootstrap self-play model)
        device: Device to train on
        output_path: Path to save final checkpoint
        logdir: Directory for Tensorboard logs
        seed: Random seed

    Returns:
        Dict with final metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    amp_context = autocast(device_type='cuda') if device_type == 'cuda' else nullcontext()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine schedule over all steps
    total_steps = n_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Mixed precision (only use scaler on CUDA)
    use_scaler = device_type == 'cuda'
    scaler = GradScaler() if use_scaler else None

    # Tensorboard
    writer = None
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)

    print(f"Training for {n_epochs} epochs ({total_steps} steps)")
    print(f"Auxiliary task: {use_auxiliary}, Value only: {value_only}")
    print(f"Parameters: {count_parameters(model):,}")

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}",
                    leave=False, ncols=80, mininterval=1.0)
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            tokens = batch['tokens'].to(device)
            mask = batch['mask'].to(device)
            value_target = batch['value'].to(device)
            policy_target = batch['policy'].to(device)
            sectors_target = batch['sectors'].to(device)

            # Forward pass
            optimizer.zero_grad()
            with amp_context:
                value_pred, policy_logits, sector_pred = model(tokens, mask)
                loss, v_loss, p_loss, s_loss = compute_losses(
                    value_pred, policy_logits, sector_pred,
                    {'value': value_target, 'policy': policy_target, 'sectors': sectors_target},
                    use_auxiliary, value_only
                )

            # Backward pass (with gradient scaling on CUDA)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Track losses
            batch_size = tokens.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size

            # Log to tensorboard
            if writer and global_step % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/value_loss', v_loss.item(), global_step)
                writer.add_scalar('train/policy_loss', p_loss.item(), global_step)
                writer.add_scalar('train/sector_loss', s_loss.item(), global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            global_step += 1

        # Epoch stats
        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / epoch_samples

        # Validation
        val_metrics = evaluate(model, val_loader, device, use_auxiliary, value_only)

        # Log validation to tensorboard
        if writer:
            writer.add_scalar('val/loss', val_metrics['loss'], global_step)
            writer.add_scalar('val/value_loss', val_metrics['value_loss'], global_step)
            writer.add_scalar('val/policy_loss', val_metrics['policy_loss'], global_step)
            writer.add_scalar('val/sector_loss', val_metrics['sector_loss'], global_step)
            writer.add_scalar('val/value_acc', val_metrics['value_acc'], global_step)
            writer.add_scalar('val/policy_acc', val_metrics['policy_acc'], global_step)

        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: v={val_metrics['value_acc']:.3f} p={val_metrics['policy_acc']:.3f}")

        # Track best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']

    # Save final checkpoint
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': n_epochs,
            'val_loss': val_metrics['loss'],
            'use_auxiliary': use_auxiliary,
            'value_only': value_only,
        }, output_path)
        print(f"Saved checkpoint to {output_path}")

    if writer:
        writer.close()

    return val_metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Domineering neural network')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to NPZ file with game data')
    parser.add_argument('--model', type=str, choices=['small', 'medium', 'large'], default='small',
                        help='Model size (small, medium, or large)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--auxiliary', action='store_true',
                        help='Use auxiliary sector task')
    parser.add_argument('--value_only', action='store_true',
                        help='Train value head only (for bootstrap self-play model)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save model checkpoint')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory for Tensorboard logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--test', action='store_true',
                        help='Run tests only')

    args = parser.parse_args()

    if args.test:
        run_training_tests()
        return

    if args.data is None:
        parser.error("--data is required when not running tests")

    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Configure DataLoader for device (CPU: fewer workers, no pinned memory)
    use_workers = args.n_workers if device == 'cuda' else min(2, args.n_workers)
    use_pin_memory = device == 'cuda'

    # Load datasets
    print(f"Loading data from {args.data}")
    train_dataset = DomineeringDataset(args.data, split='train')
    val_dataset = DomineeringDataset(args.data, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=use_workers,
        collate_fn=collate_batch,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=use_workers,
        collate_fn=collate_batch,
        pin_memory=use_pin_memory
    )

    # Create model
    if args.model == 'small':
        model = create_small_model()
    elif args.model == 'medium':
        model = create_medium_model()
    else:
        model = create_large_model()

    print(f"Created {args.model} model with {count_parameters(model):,} parameters")

    # Train
    final_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        use_auxiliary=args.auxiliary,
        value_only=args.value_only,
        device=device,
        output_path=args.output,
        logdir=args.logdir,
        seed=args.seed
    )

    print("\nFinal validation metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")


# ============================================================================
# Tests
# ============================================================================

def test_collate_batch():
    """Test collate function produces correct shapes."""
    # Create fake samples
    samples = []
    for _ in range(4):
        samples.append({
            'tokens': np.zeros(257, dtype=np.int64),
            'value': np.float32(1.0),
            'policy': np.int64(0),
            'mask': np.ones(480, dtype=bool),
            'sectors': np.zeros(16, dtype=np.float32)
        })

    batch = collate_batch(samples)

    assert batch['tokens'].shape == (4, 257), f"tokens shape: {batch['tokens'].shape}"
    assert batch['value'].shape == (4,), f"value shape: {batch['value'].shape}"
    assert batch['policy'].shape == (4,), f"policy shape: {batch['policy'].shape}"
    assert batch['mask'].shape == (4, 480), f"mask shape: {batch['mask'].shape}"
    assert batch['sectors'].shape == (4, 16), f"sectors shape: {batch['sectors'].shape}"

    print("PASS: test_collate_batch")


def test_compute_losses():
    """Test loss computation."""
    batch_size = 4
    device = 'cpu'

    # Create fake predictions
    value_pred = torch.rand(batch_size, 1)
    policy_logits = torch.randn(batch_size, 480)
    sector_pred = torch.randn(batch_size, 16)

    # Create fake targets
    batch = {
        'value': torch.randint(0, 2, (batch_size,)).float(),
        'policy': torch.randint(0, 480, (batch_size,)),
        'sectors': torch.randn(batch_size, 16)
    }

    # Test with auxiliary
    total, v, p, s = compute_losses(value_pred, policy_logits, sector_pred, batch, use_auxiliary=True)
    assert total.shape == (), f"total loss shape: {total.shape}"
    assert torch.isfinite(total), f"total loss not finite: {total}"

    # Test without auxiliary
    total_no_aux, v2, p2, s2 = compute_losses(value_pred, policy_logits, sector_pred, batch, use_auxiliary=False)
    expected = v + p
    assert torch.isclose(total_no_aux, expected), f"without auxiliary: {total_no_aux} != {expected}"

    # Test value_only mode
    total_val_only, v3, p3, s3 = compute_losses(value_pred, policy_logits, sector_pred, batch, value_only=True)
    assert torch.isclose(total_val_only, v3), f"value_only total should equal value loss"
    assert p3.item() == 0.0, f"value_only policy loss should be 0"
    assert s3.item() == 0.0, f"value_only sector loss should be 0"

    print("PASS: test_compute_losses")


def test_evaluate():
    """Test evaluation function."""
    device = 'cpu'

    # Create small model
    model = create_small_model()
    model.eval()

    # Create fake data loader
    samples = []
    for _ in range(8):
        samples.append({
            'tokens': np.random.randint(0, 2, 257).astype(np.int64),
            'value': np.float32(np.random.randint(0, 2)),
            'policy': np.int64(np.random.randint(0, 480)),
            'mask': np.ones(480, dtype=bool),
            'sectors': np.random.randn(16).astype(np.float32)
        })
        samples[-1]['tokens'][-1] = 2  # CLS token

    from torch.utils.data import DataLoader as DL
    class FakeDataset:
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    fake_loader = DL(FakeDataset(samples), batch_size=4, collate_fn=collate_batch)

    metrics = evaluate(model, fake_loader, device)

    assert 'loss' in metrics
    assert 'value_acc' in metrics
    assert 'policy_acc' in metrics
    assert 0 <= metrics['value_acc'] <= 1
    assert 0 <= metrics['policy_acc'] <= 1

    print("PASS: test_evaluate")


def test_training_step():
    """Test that a single training step works."""
    device = 'cpu'

    model = create_small_model()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create fake batch
    batch_size = 4
    tokens = torch.randint(0, 2, (batch_size, 257))
    tokens[:, -1] = 2  # CLS
    mask = torch.ones(batch_size, 480, dtype=torch.bool)
    value_target = torch.randint(0, 2, (batch_size,)).float()
    policy_target = torch.randint(0, 480, (batch_size,))
    sectors_target = torch.randn(batch_size, 16)

    # Forward pass
    optimizer.zero_grad()
    value_pred, policy_logits, sector_pred = model(tokens, mask)
    loss, _, _, _ = compute_losses(
        value_pred, policy_logits, sector_pred,
        {'value': value_target, 'policy': policy_target, 'sectors': sectors_target}
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss), f"Loss not finite: {loss}"

    print("PASS: test_training_step")


def run_training_tests():
    """Run all training tests."""
    print("=" * 60)
    print("Running Training Tests")
    print("=" * 60)

    test_collate_batch()
    test_compute_losses()
    test_evaluate()
    test_training_step()

    print("=" * 60)
    print("All training tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
