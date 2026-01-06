"""Embedding and verification for small->large network transfer.

This module implements Section 7 of the experiment:
- embed_small_into_large: Copy trained weights from small to large network
- verify_embedding: Ensure embedded layers produce identical outputs

CRITICAL: Fresh components (heads 4-7, MLP neurons 512-767) have W_O and W_out
set to exactly zero so they contribute nothing at initialization. This ensures
the embedded portion of the large network produces identical outputs to the
small network.
"""

import torch
import torch.nn as nn
from model import create_small_model, create_large_model, DomineeringTransformer
from config import SMALL_CONFIG, LARGE_CONFIG, N_TOKENS


def embed_small_into_large(small: DomineeringTransformer,
                           large: DomineeringTransformer) -> None:
    """Embed trained small network weights into a large network.

    Copies weights from small network into layers 0-1 of large network.
    Fresh heads (4-7) and fresh MLP neurons (512-767) have their output
    weights set to zero so they contribute nothing initially.

    The large network is modified in-place.

    Args:
        small: Trained small network (2 layers, 4 heads, 512 MLP neurons)
        large: Randomly initialized large network (7 layers, 8 heads, 768 MLP neurons)
    """
    # Verify dimensions are compatible
    assert small.d_model == large.d_model, f"d_model mismatch: {small.d_model} vs {large.d_model}"
    assert small.d_head == large.d_head, f"d_head mismatch: {small.d_head} vs {large.d_head}"
    assert small.n_heads <= large.n_heads, f"small has more heads than large: {small.n_heads} vs {large.n_heads}"
    assert small.n_layers <= large.n_layers, f"small has more layers than large"

    with torch.no_grad():
        # =====================================================================
        # 1. Copy embeddings (shared across all layers)
        # =====================================================================
        large.state_embed.weight.copy_(small.state_embed.weight)
        large.pos_embed.weight.copy_(small.pos_embed.weight)

        # =====================================================================
        # 2. Copy layers 0-1 (the embedded portion)
        # =====================================================================
        for layer_idx in range(small.n_layers):
            small_layer = small.layers[layer_idx]
            large_layer = large.layers[layer_idx]

            # -----------------------------------------------------------------
            # 2a. Copy attention heads 0-(n_heads_small-1), zero out the rest
            # -----------------------------------------------------------------
            small_attn = small_layer.attn
            large_attn = large_layer.attn

            # Copy matched heads (0-3 for small into large)
            for h in range(small.n_heads):
                large_attn.W_Q[h].copy_(small_attn.W_Q[h])
                large_attn.W_K[h].copy_(small_attn.W_K[h])
                large_attn.W_V[h].copy_(small_attn.W_V[h])
                large_attn.W_O[h].copy_(small_attn.W_O[h])

            # Zero out W_O for fresh heads (4-7)
            # Keep W_Q, W_K, W_V random - they'll learn new features
            for h in range(small.n_heads, large.n_heads):
                large_attn.W_O[h].zero_()

            # -----------------------------------------------------------------
            # 2b. Copy MLP neurons 0-(d_mlp_small-1), zero out the rest
            # -----------------------------------------------------------------
            small_mlp = small_layer.mlp
            large_mlp = large_layer.mlp

            d_mlp_small = small.d_mlp

            # Copy matched neurons (0-511)
            # W_in: (d_model, d_mlp) -> copy first 512 columns
            # W_out: (d_mlp, d_model) -> copy first 512 rows
            large_mlp.W_in.weight[:d_mlp_small].copy_(small_mlp.W_in.weight)
            large_mlp.W_in.bias[:d_mlp_small].copy_(small_mlp.W_in.bias)

            large_mlp.W_out.weight[:, :d_mlp_small].copy_(small_mlp.W_out.weight)
            # Note: W_out.bias is shared across all neurons, copy it fully
            large_mlp.W_out.bias.copy_(small_mlp.W_out.bias)

            # Zero out W_out columns for fresh neurons (512-767)
            # This makes fresh neurons contribute nothing to output
            large_mlp.W_out.weight[:, d_mlp_small:].zero_()

            # -----------------------------------------------------------------
            # 2c. Copy LayerNorm parameters
            # -----------------------------------------------------------------
            large_layer.ln1.weight.copy_(small_layer.ln1.weight)
            large_layer.ln1.bias.copy_(small_layer.ln1.bias)
            large_layer.ln2.weight.copy_(small_layer.ln2.weight)
            large_layer.ln2.bias.copy_(small_layer.ln2.bias)

        # =====================================================================
        # 3. Copy final LayerNorm and output heads
        # =====================================================================
        large.final_ln.weight.copy_(small.final_ln.weight)
        large.final_ln.bias.copy_(small.final_ln.bias)

        large.value_head.weight.copy_(small.value_head.weight)
        large.value_head.bias.copy_(small.value_head.bias)

        large.sector_head.weight.copy_(small.sector_head.weight)
        large.sector_head.bias.copy_(small.sector_head.bias)

        # Policy head MLP
        for i, (large_layer, small_layer) in enumerate(
            zip(large.policy_head.mlp, small.policy_head.mlp)
        ):
            if hasattr(large_layer, 'weight'):
                large_layer.weight.copy_(small_layer.weight)
                large_layer.bias.copy_(small_layer.bias)

        # Layers 2-6 of large network stay random (not copied)


def verify_embedding(small: DomineeringTransformer,
                     large: DomineeringTransformer,
                     batch: torch.Tensor,
                     tolerance: float = 1e-5) -> dict:
    """Verify that embedded layers produce identical outputs to small network.

    This is a CRITICAL check. If it fails, the embedding is incorrect and
    the experiment is invalid.

    Args:
        small: The small network (source of embedding)
        large: The large network with embedded weights
        batch: Token tensor of shape (batch_size, 257)
        tolerance: Maximum allowed absolute difference

    Returns:
        Dict with 'passed' bool and diagnostic info

    Raises:
        AssertionError if verification fails
    """
    small.eval()
    large.eval()

    with torch.no_grad():
        # Small network: full forward pass
        s_value, s_policy, s_sector = small(batch)

        # Large network: only run embedded layers (0-1)
        l_value, l_policy, l_sector = large.forward_partial(batch, n_layers=small.n_layers)

    # Compute differences
    value_diff = (s_value - l_value).abs().max().item()
    policy_diff = (s_policy - l_policy).abs().max().item()
    sector_diff = (s_sector - l_sector).abs().max().item()

    passed = (value_diff < tolerance and
              policy_diff < tolerance and
              sector_diff < tolerance)

    result = {
        'passed': passed,
        'value_diff': value_diff,
        'policy_diff': policy_diff,
        'sector_diff': sector_diff,
        'tolerance': tolerance
    }

    if not passed:
        raise AssertionError(
            f"Embedding verification FAILED!\n"
            f"  Value diff: {value_diff:.2e} (tolerance: {tolerance})\n"
            f"  Policy diff: {policy_diff:.2e} (tolerance: {tolerance})\n"
            f"  Sector diff: {sector_diff:.2e} (tolerance: {tolerance})"
        )

    return result


# ============================================================================
# Tests
# ============================================================================

def test_embed_and_verify():
    """Test that embedding works correctly with random small network."""
    print("Testing embed_small_into_large and verify_embedding...")

    # Create models
    small = create_small_model()
    large = create_large_model()

    # Embed
    embed_small_into_large(small, large)

    # Create test batch
    batch_size = 8
    tokens = torch.randint(0, 2, (batch_size, N_TOKENS))
    tokens[:, -1] = 2  # CLS token

    # Verify
    result = verify_embedding(small, large, tokens)

    print(f"  Value diff: {result['value_diff']:.2e}")
    print(f"  Policy diff: {result['policy_diff']:.2e}")
    print(f"  Sector diff: {result['sector_diff']:.2e}")
    print(f"  Passed: {result['passed']}")

    print("PASS: test_embed_and_verify")


def test_fresh_heads_zero():
    """Test that fresh attention heads have W_O = 0."""
    print("Testing that fresh heads have W_O = 0...")

    small = create_small_model()
    large = create_large_model()

    embed_small_into_large(small, large)

    # Check layers 0-1
    for layer_idx in range(small.n_layers):
        attn = large.layers[layer_idx].attn

        # Heads 0-3 should have non-zero W_O (copied from small)
        for h in range(small.n_heads):
            assert attn.W_O[h].abs().sum() > 0, f"Head {h} W_O should be non-zero"

        # Heads 4-7 should have W_O = 0
        for h in range(small.n_heads, large.n_heads):
            assert attn.W_O[h].abs().sum() == 0, f"Head {h} W_O should be zero"

    print("PASS: test_fresh_heads_zero")


def test_fresh_mlp_zero():
    """Test that fresh MLP neurons have W_out columns = 0."""
    print("Testing that fresh MLP neurons have W_out = 0...")

    small = create_small_model()
    large = create_large_model()

    embed_small_into_large(small, large)

    d_mlp_small = small.d_mlp

    # Check layers 0-1
    for layer_idx in range(small.n_layers):
        mlp = large.layers[layer_idx].mlp

        # Columns 0-511 should be non-zero (copied from small)
        assert mlp.W_out.weight[:, :d_mlp_small].abs().sum() > 0, \
            f"Layer {layer_idx} matched MLP neurons should be non-zero"

        # Columns 512-767 should be zero
        assert mlp.W_out.weight[:, d_mlp_small:].abs().sum() == 0, \
            f"Layer {layer_idx} fresh MLP neurons should be zero"

    print("PASS: test_fresh_mlp_zero")


def test_layers_2_6_random():
    """Test that layers 2-6 remain random (not touched by embedding)."""
    print("Testing that layers 2-6 remain random...")

    large_before = create_large_model()
    large_after = create_large_model()

    # Use same seed for both to compare
    torch.manual_seed(42)
    large_before = create_large_model()
    torch.manual_seed(42)
    large_after = create_large_model()

    small = create_small_model()
    embed_small_into_large(small, large_after)

    # Layers 2-6 should be identical to before (unchanged)
    for layer_idx in range(2, 7):
        before_layer = large_before.layers[layer_idx]
        after_layer = large_after.layers[layer_idx]

        # Check attention weights
        for h in range(large_before.n_heads):
            assert torch.equal(before_layer.attn.W_Q[h], after_layer.attn.W_Q[h])
            assert torch.equal(before_layer.attn.W_K[h], after_layer.attn.W_K[h])
            assert torch.equal(before_layer.attn.W_V[h], after_layer.attn.W_V[h])
            assert torch.equal(before_layer.attn.W_O[h], after_layer.attn.W_O[h])

        # Check MLP weights
        assert torch.equal(before_layer.mlp.W_in.weight, after_layer.mlp.W_in.weight)
        assert torch.equal(before_layer.mlp.W_out.weight, after_layer.mlp.W_out.weight)

    print("PASS: test_layers_2_6_random")


def run_embedding_tests():
    """Run all embedding tests."""
    print("=" * 60)
    print("Running Embedding Tests (Section 7)")
    print("=" * 60)

    test_embed_and_verify()
    test_fresh_heads_zero()
    test_fresh_mlp_zero()
    test_layers_2_6_random()

    print("=" * 60)
    print("All embedding tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_embedding_tests()
