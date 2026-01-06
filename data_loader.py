"""Data loader for Domineering training.

Reconstructs positions from move sequences with on-the-fly augmentation.
Uses deterministic game-hash-based splits to keep augmentations together.
"""

import numpy as np
import random

import domineering_game as dg

# Import constants and precomputed tables from domineering_game
N_MOVES = dg.N_MOVES
P_MOVES = dg.P_MOVES
N_SQUARES = dg.N_SQUARES
N_SECTORS = dg.N_SECTORS
BOARD_SIZE = dg.BOARD_SIZE

move_eliminations = dg.move_eliminations
move_to_squares = dg.move_to_squares
sector_v_masks = dg.sector_v_masks
sector_h_masks = dg.sector_h_masks
player_1_legal_moves = dg.player_1_legal_moves
player_2_legal_moves = dg.player_2_legal_moves

# Configuration
N_RANDOM_OPENING = 16  # Skip positions in first 16 moves (random opening)


# ============================================================================
# Core Functions
# ============================================================================

def load_games(npz_path):
    """Load game records from NPZ file.

    Returns:
        moves: (n_games, max_len) int16 array, -1 for padding
        lengths: (n_games,) int16 array of actual game lengths
        winners: (n_games,) bool array, True = vertical won
    """
    data = np.load(npz_path)
    return data['moves'], data['lengths'], data['winners']


def game_to_split(moves, length):
    """Deterministic split assignment based on game hash.

    Uses first 20 moves to hash (enough to uniquely identify game).
    Returns 'train' (80%), 'val' (10%), or 'test' (10%).
    """
    # Hash first N moves (use only valid moves, not padding)
    n_hash = min(20, length)
    h = hash(tuple(moves[:n_hash]))
    bucket = h % 10
    if bucket < 8:
        return 'train'
    elif bucket == 8:
        return 'val'
    else:
        return 'test'


def split_games(moves, lengths):
    """Split all games into train/val/test by hash.

    Returns:
        Dict with 'train', 'val', 'test' keys, each containing array of indices
    """
    n_games = len(moves)
    splits = {'train': [], 'val': [], 'test': []}
    for i in range(n_games):
        split = game_to_split(moves[i], lengths[i])
        splits[split].append(i)
    return {k: np.array(v, dtype=np.int32) for k, v in splits.items()}


def replay_to_position(moves, stop_idx, symmetry=0):
    """Replay moves to build board and remaining moves.

    Args:
        moves: Array of move indices
        stop_idx: Number of moves to replay (position is BEFORE this move)
        symmetry: 0=identity, 1=hflip, 2=vflip, 3=rot180

    Returns:
        board: (256,) bool array of occupied squares
        remaining: (480,) bool array of remaining legal moves (both players)
    """
    # Apply symmetry to move sequence
    aug_moves = dg.augment_move_sequence(moves[:stop_idx], symmetry)

    # Build board by setting squares for each move
    board = np.zeros(N_SQUARES, dtype=bool)
    remaining = np.ones(N_MOVES, dtype=bool)

    for m in aug_moves:
        # Set the two squares this move fills
        board[move_to_squares[m]] = True
        # Update remaining moves
        remaining &= move_eliminations[m]

    return board, remaining


def compute_sector_targets_fast(remaining_moves):
    """Compute all 16 sector targets in one vectorized operation.

    Args:
        remaining_moves: (480,) bool array of remaining legal moves

    Returns:
        (16,) float32 array of (v_count - h_count) / 10 per sector
        Divided by 10 to normalize to roughly [-2, 2] range for stable MSE training.
    """
    v_counts = (remaining_moves & sector_v_masks).sum(axis=1)
    h_counts = (remaining_moves & sector_h_masks).sum(axis=1)
    # Normalize by 10 to bring targets to roughly [-2, 2] range
    return ((v_counts - h_counts) / 10.0).astype(np.float32)


def make_position(moves, length, winner, pos_idx, symmetry):
    """Construct single training example matching model.py format.

    Args:
        moves: Array of move indices for this game
        length: Actual length of game (not padded)
        winner: True if vertical won
        pos_idx: Position index (0 to length-1)
        symmetry: 0=identity, 1=hflip, 2=vflip, 3=rot180

    Returns:
        Dict with tokens, value, policy, mask, sectors
    """
    board, remaining = replay_to_position(moves, pos_idx, symmetry)

    # Determine whose turn (alternates, vertical first)
    is_horizontal_turn = (pos_idx % 2 == 1)

    # Legal moves for current player
    if is_horizontal_turn:
        legal = remaining & player_2_legal_moves
    else:
        legal = remaining & player_1_legal_moves

    # Build tokens: 0/1 for board, 2 for CLS (matches model.py state_embed)
    tokens = np.zeros(257, dtype=np.int64)
    tokens[:256] = board.astype(np.int64)  # 0=empty, 1=occupied
    tokens[256] = 2  # CLS token

    # Policy target: the move actually played, augmented
    policy_target = dg.augment_move(int(moves[pos_idx]), symmetry)

    return {
        'tokens': tokens,                          # (257,) int64
        'value': np.float32(1.0 if winner else 0.0),  # float32
        'policy': np.int64(policy_target),         # int64
        'mask': legal.copy(),                      # (480,) bool
        'sectors': compute_sector_targets_fast(remaining)  # (16,) float32
    }


# ============================================================================
# Dataset Class
# ============================================================================

class DomineeringDataset:
    """PyTorch-compatible dataset for Domineering training.

    Reconstructs positions from move sequences with on-the-fly augmentation.
    Each __getitem__ samples a random position and random symmetry.
    """

    def __init__(self, npz_path, split='train', positions_per_game=10):
        """
        Args:
            npz_path: Path to NPZ file with game records
            split: 'train', 'val', or 'test'
            positions_per_game: Number of positions to sample per game per epoch
        """
        moves, lengths, winners = load_games(npz_path)
        splits = split_games(moves, lengths)

        if split not in splits:
            raise ValueError(f"Unknown split: {split}")

        indices = splits[split]
        self.moves = moves[indices]
        self.lengths = lengths[indices]
        self.winners = winners[indices]
        self.positions_per_game = positions_per_game

        print(f"DomineeringDataset: {split} split with {len(indices)} games, "
              f"{len(self)} positions per epoch")

    def __len__(self):
        return len(self.moves) * self.positions_per_game

    def __getitem__(self, idx):
        game_idx = idx // self.positions_per_game

        # Random position after opening (move 16+), before game end
        min_pos = N_RANDOM_OPENING
        max_pos = self.lengths[game_idx] - 1
        if max_pos <= min_pos:
            # Very short game, use whatever positions we have
            pos_idx = max(0, max_pos)
        else:
            pos_idx = random.randint(min_pos, max_pos)

        # Random symmetry for augmentation
        symmetry = random.randint(0, 3)

        return make_position(
            self.moves[game_idx],
            self.lengths[game_idx],
            self.winners[game_idx],
            pos_idx,
            symmetry
        )


# ============================================================================
# Tests
# ============================================================================

def test_move_to_squares():
    """Test move_to_squares lookup is correct."""
    # Vertical move 0: (0,0)-(1,0) -> squares 0, 16
    assert move_to_squares[0, 0] == 0
    assert move_to_squares[0, 1] == 16

    # Vertical move 15: (0,15)-(1,15) -> squares 15, 31
    assert move_to_squares[15, 0] == 15
    assert move_to_squares[15, 1] == 31

    # Vertical move 224: (14,0)-(15,0) -> squares 224, 240
    assert move_to_squares[224, 0] == 224
    assert move_to_squares[224, 1] == 240

    # Horizontal move 240 (k=0): j=0, i=0 -> (0,0)-(0,1) -> squares 0, 1
    assert move_to_squares[240, 0] == 0
    assert move_to_squares[240, 1] == 1

    # Horizontal move 464 (k=224): j=14, i=0 -> (0,14)-(0,15) -> squares 14, 15
    assert move_to_squares[464, 0] == 14
    assert move_to_squares[464, 1] == 15

    print("PASS: test_move_to_squares")


def test_replay_produces_valid_game():
    """Test that replaying an augmented game produces valid positions."""
    # Create a test game
    game = dg.domineering_game()
    moves = []
    for _ in range(30):
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]
        if len(legal_indices) == 0:
            break
        move = np.random.choice(legal_indices)
        moves.append(move)
        dg.make_move(game, move)

    moves = np.array(moves, dtype=np.int16)

    # Test replay at various positions with all symmetries
    for pos_idx in [5, 10, 15, len(moves) - 1]:
        if pos_idx >= len(moves):
            continue
        for symmetry in range(4):
            board, remaining = replay_to_position(moves, pos_idx, symmetry)

            # Check board has correct number of occupied squares
            expected_occupied = pos_idx * 2  # Each move fills 2 squares
            assert board.sum() == expected_occupied, \
                f"Expected {expected_occupied} occupied, got {board.sum()}"

            # Check remaining moves is consistent
            assert remaining.sum() < N_MOVES, "Should have fewer remaining moves"

    print("PASS: test_replay_produces_valid_game")


def test_sector_targets_match():
    """Test that fast sector computation matches original."""
    # Play a game and check sector targets at various positions
    game = dg.domineering_game()
    for _ in range(20):
        if game[3]:
            break
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]
        if len(legal_indices) == 0:
            break
        dg.make_move(game, np.random.choice(legal_indices))

    # Get remaining moves from game
    remaining = game[4]

    # Compare fast and original implementations
    fast_result = compute_sector_targets_fast(remaining)
    orig_result = dg.compute_sector_targets(game)

    assert np.allclose(fast_result, orig_result), \
        f"Mismatch: fast={fast_result}, orig={orig_result}"

    print("PASS: test_sector_targets_match")


def test_split_deterministic():
    """Test that splits are deterministic."""
    # Create fake moves
    np.random.seed(42)
    n_games = 100
    moves = np.random.randint(0, 480, (n_games, 50), dtype=np.int16)
    lengths = np.full(n_games, 50, dtype=np.int16)

    # Split twice
    splits1 = split_games(moves, lengths)
    splits2 = split_games(moves, lengths)

    # Should be identical
    for key in ['train', 'val', 'test']:
        assert np.array_equal(splits1[key], splits2[key]), \
            f"Split {key} not deterministic"

    # Check approximate ratios (80/10/10)
    n_train = len(splits1['train'])
    n_val = len(splits1['val'])
    n_test = len(splits1['test'])
    assert 70 < n_train < 90, f"Train ratio off: {n_train}%"
    assert 5 < n_val < 20, f"Val ratio off: {n_val}%"
    assert 5 < n_test < 20, f"Test ratio off: {n_test}%"

    # Check no overlap
    train_set = set(splits1['train'])
    val_set = set(splits1['val'])
    test_set = set(splits1['test'])
    assert len(train_set & val_set) == 0, "Train/val overlap"
    assert len(train_set & test_set) == 0, "Train/test overlap"
    assert len(val_set & test_set) == 0, "Val/test overlap"

    print("PASS: test_split_deterministic")


def test_make_position_shapes():
    """Test that make_position returns correct shapes."""
    # Create a fake game
    moves = np.array([0, 240, 16, 256, 32, 272], dtype=np.int16)
    length = 6
    winner = True

    result = make_position(moves, length, winner, pos_idx=4, symmetry=0)

    assert result['tokens'].shape == (257,), f"tokens shape: {result['tokens'].shape}"
    assert result['tokens'].dtype == np.int64
    assert result['tokens'][256] == 2, "CLS token should be 2"

    assert isinstance(result['value'], np.floating)
    assert result['value'] == 1.0

    assert isinstance(result['policy'], np.integer)
    assert 0 <= result['policy'] < N_MOVES

    assert result['mask'].shape == (480,), f"mask shape: {result['mask'].shape}"
    assert result['mask'].dtype == bool

    assert result['sectors'].shape == (16,), f"sectors shape: {result['sectors'].shape}"
    assert result['sectors'].dtype == np.float32

    print("PASS: test_make_position_shapes")


def run_data_loader_tests():
    """Run all data loader tests."""
    print("=" * 60)
    print("Running Data Loader Tests")
    print("=" * 60)

    np.random.seed(42)
    random.seed(42)

    test_move_to_squares()
    test_replay_produces_valid_game()
    test_sector_targets_match()
    test_split_deterministic()
    test_make_position_shapes()

    print("=" * 60)
    print("All data loader tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_data_loader_tests()
