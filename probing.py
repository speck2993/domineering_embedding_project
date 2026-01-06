"""Linear probing infrastructure for interpretability analysis.

Implements Section 9 of the experiment:
- Extract CLS token activations at each layer
- Train Ridge regression probes to predict sector targets
- Track probe R² throughout training

The key question: does sector information concentrate in specific layers,
and does this differ between embedded vs baseline models?
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional


def get_layer_activations(model, tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Extract CLS token activations after a specific transformer layer.

    Args:
        model: DomineeringTransformer model
        tokens: (batch, 257) token tensor
        layer_idx: Which layer to extract from (0 to n_layers-1)
                   Use -1 for after final_ln (pre-output)

    Returns:
        (batch, d_model) tensor of CLS token activations
    """
    model.eval()

    with torch.no_grad():
        # Embed tokens
        x = model.state_embed(tokens) + model.pos_embed(model.positions)

        # Run through layers up to layer_idx
        if layer_idx == -1:
            # Run all layers + final LN
            for layer in model.layers:
                x = layer(x)
            x = model.final_ln(x)
        else:
            # Run up to and including layer_idx
            for i, layer in enumerate(model.layers):
                x = layer(x)
                if i == layer_idx:
                    break

        # Extract CLS token (last position)
        cls = x[:, -1]  # (batch, d_model)

    return cls


def get_all_layer_activations(model, tokens: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Extract CLS activations from all layers at once.

    More efficient than calling get_layer_activations multiple times.

    Args:
        model: DomineeringTransformer model
        tokens: (batch, 257) token tensor

    Returns:
        Dict mapping layer_idx -> (batch, d_model) activations
        Includes -1 for post-final-LN activations
    """
    model.eval()
    activations = {}

    with torch.no_grad():
        x = model.state_embed(tokens) + model.pos_embed(model.positions)

        for i, layer in enumerate(model.layers):
            x = layer(x)
            activations[i] = x[:, -1].clone()  # CLS token

        x = model.final_ln(x)
        activations[-1] = x[:, -1].clone()

    return activations


class LinearProbe:
    """Ridge regression probe for predicting sector targets from activations."""

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Ridge regularization strength
        """
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.is_fitted = False
        self.train_r2 = None
        self.val_r2 = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit probe and optionally evaluate on validation set.

        Args:
            X_train: (n_samples, d_model) activations
            y_train: (n_samples, 16) sector targets
            X_val: Optional validation activations
            y_val: Optional validation targets
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.train_r2 = self.model.score(X_train, y_train)

        if X_val is not None and y_val is not None:
            self.val_r2 = self.model.score(X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict sector targets from activations."""
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        return self.model.score(X, y)


def train_probes_all_layers(model, dataloader, n_samples: int = 2000,
                            alpha: float = 1.0, device: str = 'cpu',
                            val_split: float = 0.2) -> Dict[int, LinearProbe]:
    """Train linear probes for all layers at once.

    Args:
        model: DomineeringTransformer model
        dataloader: DataLoader yielding batches with 'tokens' and 'sectors'
        n_samples: Max samples to collect for training
        alpha: Ridge regularization strength
        device: Device for model inference
        val_split: Fraction of samples for validation

    Returns:
        Dict mapping layer_idx -> fitted LinearProbe
        Includes -1 for post-final-LN layer
    """
    model.eval()
    model.to(device)

    # Collect activations and targets
    all_activations = {i: [] for i in range(model.n_layers)}
    all_activations[-1] = []
    all_sectors = []

    samples_collected = 0
    for batch in dataloader:
        if samples_collected >= n_samples:
            break

        tokens = batch['tokens'].to(device)
        sectors = batch['sectors'].numpy()

        # Get activations from all layers
        layer_acts = get_all_layer_activations(model, tokens)

        for layer_idx, acts in layer_acts.items():
            all_activations[layer_idx].append(acts.cpu().numpy())
        all_sectors.append(sectors)

        samples_collected += len(tokens)

    # Concatenate
    for layer_idx in all_activations:
        all_activations[layer_idx] = np.concatenate(all_activations[layer_idx], axis=0)
    all_sectors = np.concatenate(all_sectors, axis=0)

    # Train/val split
    n_total = len(all_sectors)
    n_val = int(n_total * val_split)
    indices = np.random.permutation(n_total)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    y_train = all_sectors[train_idx]
    y_val = all_sectors[val_idx]

    # Train probe for each layer
    probes = {}
    for layer_idx in all_activations:
        X = all_activations[layer_idx]
        X_train, X_val = X[train_idx], X[val_idx]

        probe = LinearProbe(alpha=alpha)
        probe.fit(X_train, y_train, X_val, y_val)
        probes[layer_idx] = probe

    return probes


def evaluate_probes(probes: Dict[int, LinearProbe]) -> Dict[str, float]:
    """Extract R² values from fitted probes.

    Args:
        probes: Dict mapping layer_idx -> fitted LinearProbe

    Returns:
        Dict with 'layer_N_r2' for each layer (train and val)
    """
    results = {}
    for layer_idx, probe in sorted(probes.items()):
        if layer_idx == -1:
            prefix = 'final'
        else:
            prefix = f'layer_{layer_idx}'

        results[f'{prefix}_train_r2'] = probe.train_r2
        results[f'{prefix}_val_r2'] = probe.val_r2

    return results


def print_probe_summary(probes: Dict[int, LinearProbe], model_name: str = "Model"):
    """Print formatted probe results."""
    print(f"\n{model_name} Probe R² (validation):")
    print("-" * 30)
    for layer_idx in sorted(probes.keys()):
        if layer_idx == -1:
            name = "Final LN"
        else:
            name = f"Layer {layer_idx}"
        probe = probes[layer_idx]
        print(f"  {name:10}: {probe.val_r2:.4f}")


# ============================================================================
# Tests
# ============================================================================

def test_get_layer_activations():
    """Test activation extraction."""
    from model import create_small_model
    from config import N_TOKENS

    model = create_small_model()
    batch_size = 4
    tokens = torch.randint(0, 2, (batch_size, N_TOKENS))
    tokens[:, -1] = 2

    # Test each layer
    for layer_idx in range(model.n_layers):
        acts = get_layer_activations(model, tokens, layer_idx)
        assert acts.shape == (batch_size, model.d_model), f"Layer {layer_idx} shape mismatch"

    # Test final LN
    acts = get_layer_activations(model, tokens, -1)
    assert acts.shape == (batch_size, model.d_model), "Final LN shape mismatch"

    print("PASS: test_get_layer_activations")


def test_get_all_layer_activations():
    """Test batch activation extraction."""
    from model import create_small_model
    from config import N_TOKENS

    model = create_small_model()
    batch_size = 4
    tokens = torch.randint(0, 2, (batch_size, N_TOKENS))
    tokens[:, -1] = 2

    all_acts = get_all_layer_activations(model, tokens)

    # Should have n_layers + 1 entries (layers 0,1 and final)
    assert len(all_acts) == model.n_layers + 1
    assert -1 in all_acts

    for layer_idx, acts in all_acts.items():
        assert acts.shape == (batch_size, model.d_model)

    print("PASS: test_get_all_layer_activations")


def test_linear_probe():
    """Test probe fitting."""
    # Synthetic data
    n_samples = 100
    d_model = 128
    n_sectors = 16

    X = np.random.randn(n_samples, d_model)
    y = np.random.randn(n_samples, n_sectors)

    # Make y slightly predictable from X
    W_true = np.random.randn(d_model, n_sectors) * 0.1
    y = X @ W_true + np.random.randn(n_samples, n_sectors) * 0.5

    # Split
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    probe = LinearProbe(alpha=1.0)
    probe.fit(X_train, y_train, X_val, y_val)

    assert probe.is_fitted
    assert probe.train_r2 is not None
    assert probe.val_r2 is not None
    assert 0 <= probe.val_r2 <= 1  # Should be reasonable for predictable data

    print(f"PASS: test_linear_probe (val R²={probe.val_r2:.4f})")


def run_probe_tests():
    """Run all probe tests."""
    print("=" * 60)
    print("Running Probe Tests (Section 9)")
    print("=" * 60)

    test_get_layer_activations()
    test_get_all_layer_activations()
    test_linear_probe()

    print("=" * 60)
    print("All probe tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_probe_tests()
