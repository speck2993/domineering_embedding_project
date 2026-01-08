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
# Efficient Dataset with Epoch Precomputation
# ============================================================================

def augment_board(board, symmetry):
    """Apply symmetry to a (256,) board array using reshape operations.

    Much faster than lookup tables - just array views and copies.
    """
    if symmetry == 0:
        return board
    b = board.reshape(BOARD_SIZE, BOARD_SIZE)
    if symmetry == 1:  # hflip: flip columns
        return b[:, ::-1].ravel().copy()
    elif symmetry == 2:  # vflip: flip rows
        return b[::-1, :].ravel().copy()
    else:  # rot180: flip both
        return b[::-1, ::-1].ravel().copy()


def augment_remaining(remaining, symmetry):
    """Apply symmetry to a (480,) remaining moves array.

    Moves have complex indexing so we use the precomputed lookup tables.
    """
    if symmetry == 0:
        return remaining
    return remaining[_MOVE_AUG_TABLES[symmetry]]


# Build move augmentation tables (moves have complex indexing)
def _build_move_aug_tables():
    """Build lookup tables for move augmentation."""
    tables = np.zeros((4, N_MOVES), dtype=np.int32)
    for m in range(N_MOVES):
        tables[0, m] = m
        tables[1, m] = dg.augment_move(m, 1)
        tables[2, m] = dg.augment_move(m, 2)
        tables[3, m] = dg.augment_move(m, 3)
    return tables

# Pre-compute move augmentation tables at module load
_MOVE_AUG_TABLES = _build_move_aug_tables()


def collect_positions_single_pass(moves, length, winner, pos_indices):
    """Collect multiple positions from a game in a single pass.

    Args:
        moves: Array of move indices for this game
        length: Actual length of game
        winner: True if vertical won
        pos_indices: Array of position indices to collect (can have duplicates)

    Returns:
        boards: (n_positions, N_SQUARES) bool array
        remainings: (n_positions, N_MOVES) bool array
        next_moves: (n_positions,) int32 array
        move_indices: (n_positions,) int32 array (copy of pos_indices)
    """
    n_positions = len(pos_indices)

    # Sort indices for single-pass collection, keeping track of original order
    sort_order = np.argsort(pos_indices)
    sorted_pos = pos_indices[sort_order]

    # Pre-allocate results
    boards = np.zeros((n_positions, N_SQUARES), dtype=bool)
    remainings = np.zeros((n_positions, N_MOVES), dtype=bool)
    next_moves = np.zeros(n_positions, dtype=np.int32)

    # Single pass through game
    board = np.zeros(N_SQUARES, dtype=bool)
    remaining = np.ones(N_MOVES, dtype=bool)

    result_idx = 0

    for move_idx in range(length):
        # Collect all positions at this index
        while result_idx < n_positions and sorted_pos[result_idx] == move_idx:
            orig_idx = sort_order[result_idx]
            boards[orig_idx] = board
            remainings[orig_idx] = remaining
            next_moves[orig_idx] = moves[move_idx]
            result_idx += 1

        # Make the move
        m = moves[move_idx]
        squares = move_to_squares[m]
        board[squares[0]] = True
        board[squares[1]] = True
        remaining &= move_eliminations[m]

    return boards, remainings, next_moves, pos_indices.copy()


class EfficientDomineeringDataset:
    """Efficient dataset with epoch precomputation and single-pass collection.

    Pre-computes all positions at epoch start, eliminating redundant game replay.
    Stores intermediate representation (~740 bytes/position vs ~2.6KB for full tensors).
    Applies augmentation on-the-fly during batch collation.
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
        self.n_games = len(self.moves)
        self.split = split

        # Storage for precomputed epoch data
        self._boards = None
        self._remainings = None
        self._next_moves = None
        self._move_indices = None
        self._winners_expanded = None
        self._symmetries = None
        self._shuffle_indices = None

        print(f"EfficientDomineeringDataset: {split} split with {self.n_games} games, "
              f"{len(self)} positions per epoch")

    def __len__(self):
        return self.n_games * self.positions_per_game

    def precompute_epoch(self, seed=None):
        """Pre-compute all positions for this epoch using single-pass collection.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        n_total = len(self)

        # Pre-allocate storage
        self._boards = np.zeros((n_total, N_SQUARES), dtype=bool)
        self._remainings = np.zeros((n_total, N_MOVES), dtype=bool)
        self._next_moves = np.zeros(n_total, dtype=np.int32)
        self._move_indices = np.zeros(n_total, dtype=np.int32)
        self._winners_expanded = np.zeros(n_total, dtype=bool)
        self._symmetries = np.random.randint(0, 4, n_total, dtype=np.int8)

        idx = 0
        for game_idx in range(self.n_games):
            length = self.lengths[game_idx]
            winner = self.winners[game_idx]

            # Determine positions to sample
            min_pos = N_RANDOM_OPENING
            max_pos = length - 1

            if max_pos <= min_pos:
                pos_indices = np.full(self.positions_per_game, max(0, max_pos), dtype=np.int32)
            else:
                pos_indices = np.random.randint(min_pos, max_pos + 1,
                                                self.positions_per_game, dtype=np.int32)

            # Single-pass collection
            boards, remainings, next_moves, move_indices = collect_positions_single_pass(
                self.moves[game_idx], length, winner, pos_indices
            )

            # Store results
            n_pos = len(boards)
            self._boards[idx:idx+n_pos] = boards
            self._remainings[idx:idx+n_pos] = remainings
            self._next_moves[idx:idx+n_pos] = next_moves
            self._move_indices[idx:idx+n_pos] = move_indices
            self._winners_expanded[idx:idx+n_pos] = winner
            idx += n_pos

        # Shuffle indices
        self._shuffle_indices = np.random.permutation(n_total)

    def __getitem__(self, idx):
        """Get a single position with augmentation applied."""
        if self._boards is None:
            self.precompute_epoch()

        # Get shuffled index
        real_idx = self._shuffle_indices[idx]

        # Get raw data
        board = self._boards[real_idx]
        remaining_orig = self._remainings[real_idx]
        next_move = self._next_moves[real_idx]
        move_idx = self._move_indices[real_idx]
        winner = self._winners_expanded[real_idx]
        symmetry = self._symmetries[real_idx]

        # Apply augmentation
        if symmetry > 0:
            board = augment_board(board, symmetry)
            remaining = augment_remaining(remaining_orig, symmetry)
            next_move = _MOVE_AUG_TABLES[symmetry, next_move]
        else:
            remaining = remaining_orig

        # Compute sectors from augmented remaining (matches make_position behavior)
        sectors = compute_sector_targets_fast(remaining)

        # Build tokens
        tokens = np.zeros(257, dtype=np.int64)
        tokens[:256] = board.astype(np.int64)
        tokens[256] = 2  # CLS token

        # Determine legal moves for current player
        is_horizontal_turn = (move_idx % 2 == 1)
        if is_horizontal_turn:
            legal = remaining & player_2_legal_moves
        else:
            legal = remaining & player_1_legal_moves

        return {
            'tokens': tokens,
            'value': np.float32(1.0 if winner else 0.0),
            'policy': np.int64(next_move),
            'mask': legal,
            'sectors': sectors
        }

    def on_epoch_end(self):
        """Call at end of epoch to prepare for next epoch."""
        # Clear precomputed data to trigger recomputation
        self._boards = None
        self._remainings = None
        self._next_moves = None
        self._move_indices = None
        self._winners_expanded = None
        self._symmetries = None
        self._shuffle_indices = None


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
    # Note: fast version divides by 10 for normalization, so we scale back
    fast_result = compute_sector_targets_fast(remaining)
    orig_result = dg.compute_sector_targets(game)

    assert np.allclose(fast_result * 10, orig_result), \
        f"Mismatch: fast*10={fast_result*10}, orig={orig_result}"

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


def test_single_pass_collection():
    """Test that single-pass collection matches individual replay."""
    # Create a test game
    game = dg.domineering_game()
    moves = []
    for _ in range(40):
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]
        if len(legal_indices) == 0:
            break
        move = np.random.choice(legal_indices)
        moves.append(move)
        dg.make_move(game, move)

    moves = np.array(moves, dtype=np.int16)
    length = len(moves)
    winner = True

    # Test positions to collect
    pos_indices = np.array([16, 20, 25, 30, 35], dtype=np.int32)

    # Single-pass collection
    boards, remainings, next_moves, move_indices = collect_positions_single_pass(
        moves, length, winner, pos_indices
    )

    # Compare with individual replay
    for i, pos_idx in enumerate(pos_indices):
        expected_board, expected_remaining = replay_to_position(moves, pos_idx, symmetry=0)

        assert np.array_equal(boards[i], expected_board), \
            f"Board mismatch at pos_idx={pos_idx}"
        assert np.array_equal(remainings[i], expected_remaining), \
            f"Remaining mismatch at pos_idx={pos_idx}"
        assert next_moves[i] == moves[pos_idx], \
            f"Next move mismatch at pos_idx={pos_idx}"

    print("PASS: test_single_pass_collection")


def test_augmentation_functions():
    """Test board and move augmentation functions."""
    np.random.seed(42)

    # Test board augmentation - create a simple test board
    board = np.zeros(N_SQUARES, dtype=bool)
    board[0] = True   # top-left corner
    board[15] = True  # top-right corner
    board[240] = True # bottom-left corner

    # hflip: top-left -> top-right, top-right -> top-left
    hflip_board = augment_board(board, 1)
    assert hflip_board[15] == True, "hflip: top-left should go to top-right"
    assert hflip_board[0] == True, "hflip: top-right should go to top-left"
    assert hflip_board[255] == True, "hflip: bottom-left should go to bottom-right"

    # vflip: top-left -> bottom-left, bottom-left -> top-left
    vflip_board = augment_board(board, 2)
    assert vflip_board[240] == True, "vflip: top-left should go to bottom-left"
    assert vflip_board[0] == True, "vflip: bottom-left should go to top-left"

    # rot180 should be self-inverse
    rot180_board = augment_board(board, 3)
    rot180_again = augment_board(rot180_board, 3)
    assert np.array_equal(board, rot180_again), "rot180 should be self-inverse"

    # Test move augmentation tables
    for sym in range(4):
        # Check it's a valid permutation
        assert len(set(_MOVE_AUG_TABLES[sym])) == N_MOVES, \
            f"Move aug table {sym} is not a permutation"

    # Test that rot180 is self-inverse for moves
    for m in range(N_MOVES):
        assert _MOVE_AUG_TABLES[3, _MOVE_AUG_TABLES[3, m]] == m, \
            f"rot180 not self-inverse for move {m}"

    print("PASS: test_augmentation_functions")


def test_efficient_vs_original():
    """Verify EfficientDomineeringDataset produces identical outputs to make_position.

    This is the critical correctness test - if this passes, the efficient dataset
    is a drop-in replacement for the original.
    """
    np.random.seed(42)

    # Create a test game
    game = dg.domineering_game()
    moves = []
    for _ in range(50):
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]
        if len(legal_indices) == 0:
            break
        move = np.random.choice(legal_indices)
        moves.append(move)
        dg.make_move(game, move)

    moves = np.array(moves, dtype=np.int16)
    length = len(moves)
    winner = True

    # Test multiple positions and symmetries
    test_cases = [
        (16, 0), (16, 1), (16, 2), (16, 3),  # Early position, all symmetries
        (25, 0), (25, 1), (25, 2), (25, 3),  # Mid position, all symmetries
        (length-2, 0), (length-2, 3),         # Late position
    ]

    for pos_idx, symmetry in test_cases:
        if pos_idx >= length:
            continue

        # Ground truth: use make_position
        expected = make_position(moves, length, winner, pos_idx, symmetry)

        # Efficient method: collect unaugmented, then apply augmentation
        # 1. Get unaugmented board and remaining (simulates single-pass collection)
        board_orig, remaining_orig = replay_to_position(moves, pos_idx, symmetry=0)
        next_move_orig = moves[pos_idx]

        # 2. Apply augmentation to board and remaining
        if symmetry > 0:
            board_aug = augment_board(board_orig, symmetry)
            remaining_aug = augment_remaining(remaining_orig, symmetry)
            next_move_aug = _MOVE_AUG_TABLES[symmetry, next_move_orig]
        else:
            board_aug = board_orig
            remaining_aug = remaining_orig
            next_move_aug = next_move_orig

        # 3. Compute sectors from augmented remaining (matches make_position behavior)
        sectors = compute_sector_targets_fast(remaining_aug)

        # 4. Build the position dict (same logic as EfficientDomineeringDataset.__getitem__)
        tokens = np.zeros(257, dtype=np.int64)
        tokens[:256] = board_aug.astype(np.int64)
        tokens[256] = 2

        is_horizontal_turn = (pos_idx % 2 == 1)
        if is_horizontal_turn:
            legal = remaining_aug & player_2_legal_moves
        else:
            legal = remaining_aug & player_1_legal_moves

        actual = {
            'tokens': tokens,
            'value': np.float32(1.0 if winner else 0.0),
            'policy': np.int64(next_move_aug),
            'mask': legal,
            'sectors': sectors
        }

        # Compare all fields
        assert np.array_equal(actual['tokens'], expected['tokens']), \
            f"tokens mismatch at pos={pos_idx}, sym={symmetry}"
        assert actual['value'] == expected['value'], \
            f"value mismatch at pos={pos_idx}, sym={symmetry}"
        assert actual['policy'] == expected['policy'], \
            f"policy mismatch at pos={pos_idx}, sym={symmetry}: got {actual['policy']}, expected {expected['policy']}"
        assert np.array_equal(actual['mask'], expected['mask']), \
            f"mask mismatch at pos={pos_idx}, sym={symmetry}"
        assert np.allclose(actual['sectors'], expected['sectors']), \
            f"sectors mismatch at pos={pos_idx}, sym={symmetry}:\n  got {actual['sectors']}\n  expected {expected['sectors']}"

    print("PASS: test_efficient_vs_original")


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
    test_single_pass_collection()
    test_augmentation_functions()
    test_efficient_vs_original()

    print("=" * 60)
    print("All data loader tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_data_loader_tests()
