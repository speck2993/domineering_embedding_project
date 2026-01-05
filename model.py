import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SMALL_CONFIG, LARGE_CONFIG, MEDIUM_CONFIG, N_TOKENS, N_STATES, N_POLICY, N_SECTORS

# Move indexing constants (from domineering_game.py)
BOARD_SIZE = 16
P_MOVES = 240  # Vertical moves are 0-239, horizontal are 240-479


class MultiHeadAttention(nn.Module):
    """Multi-head attention with per-head weight storage.

    CRITICAL: Weights are stored as ParameterList of per-head matrices,
    NOT concatenated. This is essential for the embedding mechanism in Section 7.
    """

    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5

        # Per-head weight matrices (NOT concatenated)
        # Shape per head: W_Q, W_K, W_V are (d_model, d_head), W_O is (d_head, d_model)
        self.W_Q = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_head) * d_model**-0.5)
            for _ in range(n_heads)
        ])
        self.W_K = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_head) * d_model**-0.5)
            for _ in range(n_heads)
        ])
        self.W_V = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_head) * d_model**-0.5)
            for _ in range(n_heads)
        ])
        self.W_O = nn.ParameterList([
            nn.Parameter(torch.randn(d_head, d_model) * d_head**-0.5)
            for _ in range(n_heads)
        ])

    def forward(self, x):
        # x: (batch, seq, d_model)
        out = torch.zeros_like(x)

        for h in range(self.n_heads):
            Q = x @ self.W_Q[h]  # (batch, seq, d_head)
            K = x @ self.W_K[h]  # (batch, seq, d_head)
            V = x @ self.W_V[h]  # (batch, seq, d_head)

            # Scaled dot-product attention
            attn = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)  # (batch, seq, seq)
            head_out = (attn @ V) @ self.W_O[h]  # (batch, seq, d_model)
            out = out + head_out

        return out


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_in, d_hidden, d_out=None):
        super().__init__()
        if d_out is None:
            d_out = d_in
        self.W_in = nn.Linear(d_in, d_hidden)
        self.W_out = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.W_out(F.gelu(self.W_in(x)))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block.

    Structure: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """

    def __init__(self, d_model, n_heads, d_head, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, d_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PolicyHead(nn.Module):
    """Policy head that predicts move logits from spatial token pairs.

    For each move, concatenates embeddings of the two squares involved
    and passes through an MLP to get a single logit.
    """

    def __init__(self, d_model, d_hidden=None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_model
        # MLP: 2*d_model -> d_hidden -> 1
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1)
        )

        # Import move-to-square indices from domineering_game
        import domineering_game as dg
        move_to_squares = dg.move_to_squares  # (480, 2) array

        # Register as buffers for automatic device transfer (must be long for indexing)
        self.register_buffer('sq1_indices', torch.tensor(move_to_squares[:, 0], dtype=torch.long))
        self.register_buffer('sq2_indices', torch.tensor(move_to_squares[:, 1], dtype=torch.long))

    def forward(self, spatial_embeddings):
        """
        Args:
            spatial_embeddings: (batch, 256, d_model) - embeddings for spatial tokens only

        Returns:
            logits: (batch, 480) - one logit per move
        """
        batch_size = spatial_embeddings.shape[0]

        # Gather embeddings for each move's squares
        # sq1_embeds, sq2_embeds: (batch, 480, d_model)
        sq1_embeds = spatial_embeddings[:, self.sq1_indices]
        sq2_embeds = spatial_embeddings[:, self.sq2_indices]

        # Concatenate: (batch, 480, 2*d_model)
        paired = torch.cat([sq1_embeds, sq2_embeds], dim=-1)

        # MLP to get logits: (batch, 480, 1) -> (batch, 480)
        logits = self.mlp(paired).squeeze(-1)

        return logits


class DomineeringTransformer(nn.Module):
    """Transformer for Domineering game.

    Input: 257 tokens (256 spatial + 1 CLS)
    Output: value (scalar), policy (480 logits), sector (16 values)

    Policy is computed from spatial token pairs (not CLS).
    Value and sector are computed from CLS token.
    """

    def __init__(self, d_model, n_heads, d_head, n_layers, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.d_mlp = d_mlp

        # Embeddings
        self.state_embed = nn.Embedding(N_STATES, d_model)  # 0=empty, 1=occupied, 2=CLS
        self.pos_embed = nn.Embedding(N_TOKENS, d_model)    # 257 positions

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_head, d_mlp)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Output heads
        self.value_head = nn.Linear(d_model, 1)      # From CLS
        self.sector_head = nn.Linear(d_model, N_SECTORS)  # From CLS
        self.policy_head = PolicyHead(d_model, d_model)   # From spatial pairs

        # Register position indices as buffer
        self.register_buffer('positions', torch.arange(N_TOKENS))

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens: (batch, 257) integer tensor with values 0, 1, or 2
            mask: (batch, 480) boolean tensor - True for legal moves

        Returns:
            value: (batch, 1) sigmoid output for game outcome
            policy_logits: (batch, 480) move logits (masked if mask provided)
            sector: (batch, 16) sector balance predictions
        """
        # Embed tokens and add positional embeddings
        x = self.state_embed(tokens) + self.pos_embed(self.positions)  # (batch, 257, d_model)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)

        # Split spatial tokens and CLS token
        spatial = x[:, :256]  # (batch, 256, d_model)
        cls = x[:, -1]        # (batch, d_model)

        # Output heads
        value = self.value_head(cls)  # (batch, 1) - logits, apply sigmoid outside for prob
        sector = self.sector_head(cls)               # (batch, 16)
        policy_logits = self.policy_head(spatial)    # (batch, 480)

        # Mask illegal moves
        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e4)

        return value, policy_logits, sector

    def forward_partial(self, tokens, n_layers, mask=None):
        """Run forward pass through only the first n_layers.

        Used for verify_embedding to check that embedded layers produce
        identical outputs to the small network.

        Args:
            tokens: (batch, 257) integer tensor with values 0, 1, or 2
            n_layers: Number of transformer layers to run (e.g., 2 for small network depth)
            mask: (batch, 480) boolean tensor - True for legal moves

        Returns:
            value: (batch, 1) sigmoid output for game outcome
            policy_logits: (batch, 480) move logits (masked if mask provided)
            sector: (batch, 16) sector balance predictions
        """
        # Embed tokens and add positional embeddings
        x = self.state_embed(tokens) + self.pos_embed(self.positions)

        # Pass through only first n_layers
        for layer in self.layers[:n_layers]:
            x = layer(x)
        x = self.final_ln(x)

        # Split spatial tokens and CLS token
        spatial = x[:, :256]
        cls = x[:, -1]

        # Output heads
        value = self.value_head(cls)
        sector = self.sector_head(cls)
        policy_logits = self.policy_head(spatial)

        # Mask illegal moves
        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e4)

        return value, policy_logits, sector


def create_small_model():
    """Create small model (4 heads, 2 layers)."""
    return DomineeringTransformer(**SMALL_CONFIG)


def create_medium_model():
    """Create medium model for bootstrap selfplay (6 heads, 7 layers, d_model=96)."""
    return DomineeringTransformer(**MEDIUM_CONFIG)


def create_large_model():
    """Create large model (8 heads, 7 layers)."""
    return DomineeringTransformer(**LARGE_CONFIG)


def count_parameters(model):
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters())


def game_to_tokens(game):
    """Convert game state to token tensor.

    Args:
        game: Game state from domineering_game.py
              game[0] is (16, 16) boolean board

    Returns:
        (257,) numpy array with values 0 (empty), 1 (occupied), 2 (CLS)
    """
    # Flatten board: False->0, True->1
    spatial = game[0].flatten().astype(np.int64)  # 256 values
    cls = np.array([2], dtype=np.int64)           # CLS token
    return np.concatenate([spatial, cls])          # shape (257,)


def batch_games_to_tokens(games):
    """Convert multiple games to batched token tensor.

    Args:
        games: List of game states

    Returns:
        torch.LongTensor of shape (batch, 257)
    """
    tokens = np.stack([game_to_tokens(g) for g in games])
    return torch.from_numpy(tokens)


# ============================================================================
# Section 2.5 Tests
# ============================================================================

def test_forward_pass():
    """Test that forward pass works without errors."""
    small = create_small_model()
    large = create_large_model()

    # Random batch of tokens
    batch_size = 4
    tokens = torch.randint(0, 2, (batch_size, N_TOKENS))
    tokens[:, -1] = 2  # CLS token

    # Forward pass - small
    v, p, s = small(tokens)
    assert v.shape == (batch_size, 1), f"Small value shape: {v.shape}"
    assert p.shape == (batch_size, N_POLICY), f"Small policy shape: {p.shape}"
    assert s.shape == (batch_size, N_SECTORS), f"Small sector shape: {s.shape}"

    # Forward pass - large
    v, p, s = large(tokens)
    assert v.shape == (batch_size, 1), f"Large value shape: {v.shape}"
    assert p.shape == (batch_size, N_POLICY), f"Large policy shape: {p.shape}"
    assert s.shape == (batch_size, N_SECTORS), f"Large sector shape: {s.shape}"

    print("PASS: test_forward_pass")


def test_output_shapes():
    """Test output shapes match specification."""
    model = create_small_model()

    for batch_size in [1, 8, 32]:
        tokens = torch.randint(0, 2, (batch_size, N_TOKENS))
        tokens[:, -1] = 2

        v, p, s = model(tokens)

        assert v.shape == (batch_size, 1), f"Value shape mismatch for batch {batch_size}"
        assert p.shape == (batch_size, N_POLICY), f"Policy shape mismatch for batch {batch_size}"
        assert s.shape == (batch_size, N_SECTORS), f"Sector shape mismatch for batch {batch_size}"

        # Value should be in [0, 1] due to sigmoid
        assert (v >= 0).all() and (v <= 1).all(), "Value not in [0, 1]"

    print("PASS: test_output_shapes")


def test_masking():
    """Test that illegal move masking works correctly."""
    model = create_small_model()

    batch_size = 4
    tokens = torch.randint(0, 2, (batch_size, N_TOKENS))
    tokens[:, -1] = 2

    # Create mask where only moves 0, 10, 100 are legal
    mask = torch.zeros(batch_size, N_POLICY, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 10] = True
    mask[:, 100] = True

    # Forward with mask
    _, policy_logits, _ = model(tokens, mask=mask)

    # Check that illegal moves have very negative values
    illegal_mask = ~mask
    illegal_logits = policy_logits[illegal_mask]
    assert (illegal_logits < -1e8).all(), "Illegal moves not properly masked"

    # Check that legal moves are finite
    legal_logits = policy_logits[mask]
    assert torch.isfinite(legal_logits).all(), "Legal moves have non-finite values"

    print("PASS: test_masking")


def test_param_counts():
    """Test and report parameter counts."""
    small = create_small_model()
    large = create_large_model()

    small_params = count_parameters(small)
    large_params = count_parameters(large)

    print(f"Small model parameters: {small_params:,}")
    print(f"Large model parameters: {large_params:,}")

    # Sanity checks - should be in reasonable ballpark
    assert small_params > 100_000, f"Small model too small: {small_params}"
    assert small_params < 1_000_000, f"Small model too large: {small_params}"
    assert large_params > 1_000_000, f"Large model too small: {large_params}"
    assert large_params < 5_000_000, f"Large model too large: {large_params}"

    print("PASS: test_param_counts")


def test_game_to_tokens():
    """Test game state conversion to tokens."""
    # Import here to avoid circular dependency during module load
    import domineering_game as dg

    game = dg.domineering_game()
    tokens = game_to_tokens(game)

    assert tokens.shape == (N_TOKENS,), f"Token shape: {tokens.shape}"
    assert tokens[-1] == 2, "CLS token should be 2"
    assert (tokens[:-1] <= 1).all(), "Spatial tokens should be 0 or 1"

    # Empty board should have all zeros for spatial tokens
    assert (tokens[:-1] == 0).all(), "Empty board should have all 0s"

    # Make a move and check token updates
    dg.make_move(game, 0)  # Vertical move at (0,0)-(1,0)
    tokens = game_to_tokens(game)

    # Positions (0,0) and (1,0) should now be 1
    assert tokens[0] == 1, "Position (0,0) should be occupied"
    assert tokens[16] == 1, "Position (1,0) should be occupied"

    print("PASS: test_game_to_tokens")


def test_policy_head_indices():
    """Test that policy head correctly maps moves to square pairs."""
    policy_head = PolicyHead(d_model=128)

    # Check a few vertical moves
    # Move 0: vertical at (0,0)-(1,0) -> squares 0 and 16
    assert policy_head.sq1_indices[0].item() == 0
    assert policy_head.sq2_indices[0].item() == 16

    # Move 1: vertical at (0,1)-(1,1) -> squares 1 and 17
    assert policy_head.sq1_indices[1].item() == 1
    assert policy_head.sq2_indices[1].item() == 17

    # Check a few horizontal moves
    # Horizontal move indexing: j, i = divmod(move - 240, 16)
    # Places domino at (i, j) and (i, j+1)

    # Move 240 (k=0): j=0, i=0 -> (0,0)-(0,1) -> squares 0 and 1
    assert policy_head.sq1_indices[240].item() == 0
    assert policy_head.sq2_indices[240].item() == 1

    # Move 241 (k=1): j=0, i=1 -> (1,0)-(1,1) -> squares 16 and 17
    assert policy_head.sq1_indices[241].item() == 16
    assert policy_head.sq2_indices[241].item() == 17

    # Move 256 (k=16): j=1, i=0 -> (0,1)-(0,2) -> squares 1 and 2
    assert policy_head.sq1_indices[256].item() == 1
    assert policy_head.sq2_indices[256].item() == 2

    print("PASS: test_policy_head_indices")


def test_forward_partial():
    """Test that forward_partial produces same output as forward for small network."""
    small = create_small_model()
    large = create_large_model()

    batch_size = 4
    tokens = torch.randint(0, 2, (batch_size, N_TOKENS))
    tokens[:, -1] = 2

    # For small network, forward_partial(n_layers=2) should equal forward()
    v1, p1, s1 = small(tokens)
    v2, p2, s2 = small.forward_partial(tokens, n_layers=2)

    assert torch.allclose(v1, v2), "Small model forward vs forward_partial mismatch (value)"
    assert torch.allclose(p1, p2), "Small model forward vs forward_partial mismatch (policy)"
    assert torch.allclose(s1, s2), "Small model forward vs forward_partial mismatch (sector)"

    # For large network, forward_partial(n_layers=2) should differ from forward()
    # (because layers 2-6 exist and are random)
    v3, p3, s3 = large(tokens)
    v4, p4, s4 = large.forward_partial(tokens, n_layers=2)

    # These should NOT be equal (layers 2-6 are random and contribute)
    # We just check that forward_partial runs without error and produces valid shapes
    assert v4.shape == (batch_size, 1), f"Large partial value shape: {v4.shape}"
    assert p4.shape == (batch_size, N_POLICY), f"Large partial policy shape: {p4.shape}"
    assert s4.shape == (batch_size, N_SECTORS), f"Large partial sector shape: {s4.shape}"

    print("PASS: test_forward_partial")


def run_section_2_tests():
    """Run all Section 2.5 tests."""
    print("=" * 60)
    print("Running Section 2.5 Tests")
    print("=" * 60)

    test_forward_pass()
    test_output_shapes()
    test_masking()
    test_param_counts()
    test_game_to_tokens()
    test_policy_head_indices()
    test_forward_partial()

    print("=" * 60)
    print("All Section 2.5 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_section_2_tests()
