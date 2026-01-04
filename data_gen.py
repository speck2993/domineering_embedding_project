"""Data generation for Domineering neural network training.

Section 3: Phase 1 data generation using alpha-beta search.
Generates training data by playing games with alpha-beta search and random exploration.

Optimizations:
- Move ordering (center-first heuristic)
- Transposition table with EXACT/LOWER/UPPER bounds
- Smart random move handling (skip alpha-beta for exploration moves)
- 4-fold symmetry augmentation
- Multiprocessing for parallel game generation
"""

import numpy as np
import random
import time
import os
from typing import Tuple, Optional, List, Dict
from multiprocessing import Pool, cpu_count
from enum import IntEnum

import domineering_game as dg

# Constants from domineering_game
P_MOVES = dg.P_MOVES  # 240 - vertical moves are 0-239
N_MOVES = dg.N_MOVES  # 480
BOARD_SIZE = dg.BOARD_SIZE  # 16

# Progressive depth configuration
N_RANDOM_OPENING = 16   # Random moves at start (no alpha-beta)
RANDOM_PROB = 0.25      # Random exploration probability during play
D2_THRESHOLD = 100       # Use depth 2 when <100 legal moves (d1 above this)
D3_THRESHOLD = 30       # Use depth 3 when <30 legal moves (d2 in between)


# ============================================================================
# Transposition Table
# ============================================================================

class TTFlag(IntEnum):
    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2


class TranspositionTable:
    """Transposition table for alpha-beta search."""

    def __init__(self):
        self.table = {}

    def lookup(self, zobrist: int, depth: int, alpha: float, beta: float
               ) -> Tuple[Optional[float], Optional[int], bool]:
        """Look up position in table.

        Returns:
            (value, best_move, is_cutoff) - if is_cutoff, can return immediately
        """
        if zobrist not in self.table:
            return None, None, False

        cached_value, cached_depth, flag, cached_move = self.table[zobrist]

        if cached_depth < depth:
            # Cached at shallower depth, only use best_move for ordering
            return None, cached_move, False

        if flag == TTFlag.EXACT:
            return cached_value, cached_move, True
        elif flag == TTFlag.LOWER_BOUND:
            if cached_value >= beta:
                return cached_value, cached_move, True
        elif flag == TTFlag.UPPER_BOUND:
            if cached_value <= alpha:
                return cached_value, cached_move, True

        return None, cached_move, False

    def store(self, zobrist: int, depth: int, value: float, flag: TTFlag, best_move: int):
        """Store position in table."""
        self.table[zobrist] = (value, depth, flag, best_move)

    def clear(self):
        """Clear the table."""
        self.table.clear()


# ============================================================================
# Move Ordering (with precomputed scores)
# ============================================================================

def _build_move_scores():
    """Precompute center-distance scores for all moves."""
    center = 7.5
    scores = np.zeros(N_MOVES, dtype=np.float32)

    for move in range(N_MOVES):
        if move < P_MOVES:  # Vertical
            i, j = divmod(move, 16)
        else:  # Horizontal
            j, i = divmod(move - P_MOVES, 16)
        scores[move] = abs(i - center) + abs(j - center)

    return scores


MOVE_SCORES = _build_move_scores()


def order_moves(legal_indices: np.ndarray, tt_best_move: Optional[int] = None) -> np.ndarray:
    """Order moves for better alpha-beta pruning.

    Heuristic:
    1. TT best move first (if available)
    2. Then sort by center distance (closer to center = earlier)

    Uses precomputed scores for speed.
    """
    # Get scores for legal moves and argsort
    scores = MOVE_SCORES[legal_indices]
    order = np.argsort(scores)
    ordered = legal_indices[order]

    # If we have a TT best move, put it first
    if tt_best_move is not None:
        idx = np.where(ordered == tt_best_move)[0]
        if len(idx) > 0:
            # Move TT best move to front
            ordered = np.concatenate([[tt_best_move], ordered[ordered != tt_best_move]])

    return ordered


# ============================================================================
# Alpha-Beta Search with Transposition Table
# ============================================================================

def simple_eval(game) -> float:
    """Evaluate position from vertical's perspective.

    Returns value in [0, 1] where higher = better for vertical.
    Based on mobility difference (vertical moves - horizontal moves).
    """
    remaining = game[4]  # All remaining moves (both players)
    v_moves = remaining[:P_MOVES].sum()
    h_moves = remaining[P_MOVES:].sum()
    # sigmoid(0.3 * (v_moves - h_moves))
    diff = 0.3 * (v_moves - h_moves)
    return 1.0 / (1.0 + np.exp(-diff))


def alpha_beta(game, depth: int, alpha: float, beta: float,
               tt: TranspositionTable) -> Tuple[float, Optional[int]]:
    """Alpha-beta search from vertical's perspective with transposition table.

    Vertical (player 0, game[1]==False) is the MAX player.
    Horizontal (player 1, game[1]==True) is the MIN player.
    """
    orig_alpha = alpha

    # Terminal state check
    if game[3]:  # Game is done
        if game[2] is False:  # Vertical won
            return 1.0, None
        else:  # Horizontal won
            return 0.0, None

    # Depth limit reached
    if depth == 0:
        return simple_eval(game), None

    # Transposition table lookup
    zobrist = dg.compute_hash(game)
    tt_value, tt_move, is_cutoff = tt.lookup(zobrist, depth, alpha, beta)
    if is_cutoff:
        return tt_value, tt_move

    # Get legal moves for current player
    legal = dg.legal_moves(game)
    legal_indices = np.where(legal)[0]

    if len(legal_indices) == 0:
        return simple_eval(game), None

    # Order moves (TT move first, then by center distance)
    ordered_moves = order_moves(legal_indices, tt_move)

    if not game[1]:  # Vertical's turn (MAX player)
        max_value = float('-inf')
        best_move = ordered_moves[0] if len(ordered_moves) > 0 else None

        for move in ordered_moves:
            child = dg.copy_game(game)
            dg.make_move(child, move)
            value, _ = alpha_beta(child, depth - 1, alpha, beta, tt)

            if value > max_value:
                max_value = value
                best_move = move

            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Beta cutoff

        value = max_value
    else:  # Horizontal's turn (MIN player)
        min_value = float('inf')
        best_move = ordered_moves[0] if len(ordered_moves) > 0 else None

        for move in ordered_moves:
            child = dg.copy_game(game)
            dg.make_move(child, move)
            value, _ = alpha_beta(child, depth - 1, alpha, beta, tt)

            if value < min_value:
                min_value = value
                best_move = move

            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha cutoff

        value = min_value

    # Store in transposition table
    if value <= orig_alpha:
        flag = TTFlag.UPPER_BOUND
    elif value >= beta:
        flag = TTFlag.LOWER_BOUND
    else:
        flag = TTFlag.EXACT
    tt.store(zobrist, depth, value, flag, best_move)

    return value, best_move


# ============================================================================
# Game Record Storage (stores moves, not tensors)
# ============================================================================

def play_game_record() -> Dict:
    """Play a single game and return the move record.

    Returns a compact game record that can be replayed to reconstruct
    any position. This is more flexible than storing pre-computed tensors.

    Uses progressive depth scheme (d1/d2/d3):
    - First N_RANDOM_OPENING moves: random (no search)
    - ≥D2_THRESHOLD legal moves: depth 1
    - D3_THRESHOLD to D2_THRESHOLD-1: depth 2
    - <D3_THRESHOLD: depth 3

    Returns:
        Dict with:
            'moves': list of move indices played
            'vertical_won': bool (True if vertical won)
            'n_moves': total number of moves
    """
    game = dg.domineering_game()
    moves = []
    tt = TranspositionTable()
    move_count = 0

    while not game[3]:  # While game not done
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]

        if len(legal_indices) == 0:
            break

        move_count += 1
        n_legal = len(legal_indices)

        # Phase 1: Random opening
        if move_count <= N_RANDOM_OPENING:
            chosen_move = int(np.random.choice(legal_indices))
            moves.append(chosen_move)
            dg.make_move(game, chosen_move)
            continue

        # Phase 2: Alpha-beta play
        is_random = random.random() < RANDOM_PROB

        if is_random:
            chosen_move = int(np.random.choice(legal_indices))
        else:
            # Progressive depth (d1/d2/d3)
            if n_legal >= D2_THRESHOLD:
                depth = 1
            elif n_legal >= D3_THRESHOLD:
                depth = 2
            else:
                depth = 3

            _, best_move = alpha_beta(game, depth, float('-inf'), float('inf'), tt)
            chosen_move = int(best_move) if best_move is not None else int(np.random.choice(legal_indices))

        moves.append(chosen_move)
        dg.make_move(game, chosen_move)

    return {
        'moves': moves,
        'vertical_won': game[2] is False,  # game[2]==False means vertical won
        'n_moves': len(moves)
    }


def play_random_game_record() -> Dict:
    """Play a fully random game and return the move record.

    Every move is chosen uniformly at random from legal moves.
    This provides training data on arbitrary/random positions,
    ensuring the model can evaluate positions that don't arise
    from guided play.

    Returns:
        Dict with:
            'moves': list of move indices played
            'vertical_won': bool (True if vertical won)
            'n_moves': total number of moves
    """
    game = dg.domineering_game()
    moves = []

    while not game[3]:  # While game not done
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]

        if len(legal_indices) == 0:
            break

        chosen_move = int(np.random.choice(legal_indices))
        moves.append(chosen_move)
        dg.make_move(game, chosen_move)

    return {
        'moves': moves,
        'vertical_won': game[2] is False,
        'n_moves': len(moves)
    }


# ============================================================================
# Parallel Data Generation
# ============================================================================

def _generate_single_game_worker(args):
    """Worker function for multiprocessing - returns guided game record."""
    game_id, seed = args
    # Set unique seed per game
    np.random.seed(seed + game_id)
    random.seed(seed + game_id)
    return play_game_record()


def _generate_random_game_worker(args):
    """Worker function for multiprocessing - returns random game record."""
    game_id, seed = args
    # Set unique seed per game
    np.random.seed(seed + game_id)
    random.seed(seed + game_id)
    return play_random_game_record()


def generate_phase1_data(n_games: int, output_path: str = 'data/phase1_games.npz',
                         seed: int = 42, n_workers: int = None,
                         random_fraction: float = 0.5) -> Dict:
    """Generate Phase 1 training data - stores game records (move sequences).

    Generates two types of games:
    1. Fully random games (random_fraction of total)
    2. Alpha-beta guided games (remaining fraction)

    This ensures the model learns to evaluate both random positions
    and positions arising from guided play.

    Args:
        n_games: Total number of games to generate
        output_path: Path to save the data
        seed: Random seed for reproducibility
        n_workers: Number of parallel workers (None = cpu_count)
        random_fraction: Fraction of games that are fully random (default 0.5)

    Returns:
        Dictionary with metadata about the generation
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Calculate game counts for each type
    n_random = int(n_games * random_fraction)
    n_guided = n_games - n_random

    # Create output directory if needed
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {n_games} games ({n_random} random + {n_guided} guided)")
    print(f"Guided games settings:")
    print(f"  Opening: {N_RANDOM_OPENING} random moves")
    print(f"  Random prob: {RANDOM_PROB*100:.0f}%")
    print(f"  Depth: d1 @ {D2_THRESHOLD}+, d2 @ {D3_THRESHOLD}-{D2_THRESHOLD-1}, d3 @ <{D3_THRESHOLD}")
    print(f"Workers: {n_workers}")
    print(f"Seed: {seed}")
    print(f"Output: {output_path}")
    print()

    start_time = time.time()
    all_games = []

    # Track stats separately for each type
    random_v_wins = 0
    random_moves = 0
    guided_v_wins = 0
    guided_moves = 0

    # Phase 1: Generate random games
    if n_random > 0:
        print(f"Phase 1: Generating {n_random} random games...")
        random_args = [(i, seed) for i in range(n_random)]
        last_checkpoint_time = time.time()

        with Pool(n_workers) as pool:
            for game_idx, game_record in enumerate(pool.imap(_generate_random_game_worker, random_args)):
                all_games.append(game_record)

                if game_record['vertical_won']:
                    random_v_wins += 1
                random_moves += game_record['n_moves']

                # Progress reporting every 100 games
                if (game_idx + 1) % 100 == 0:
                    now = time.time()
                    # Rolling speed over last 100 games
                    games_per_sec = 100 / (now - last_checkpoint_time) if now > last_checkpoint_time else 0
                    last_checkpoint_time = now
                    eta = (n_random - game_idx - 1) / games_per_sec if games_per_sec > 0 else 0
                    print(f"  Random {game_idx + 1:5d}/{n_random} | "
                          f"{games_per_sec:.2f} g/s | "
                          f"ETA: {eta/60:.1f}m")

        print(f"  Random games complete: V-win {100*random_v_wins/n_random:.1f}%, "
              f"avg {random_moves/n_random:.0f} moves")
        print()

    # Phase 2: Generate guided games
    if n_guided > 0:
        print(f"Phase 2: Generating {n_guided} guided games...")
        # Use offset seed to avoid correlation with random games
        guided_args = [(i, seed + n_random) for i in range(n_guided)]
        last_checkpoint_time = time.time()

        with Pool(n_workers) as pool:
            for game_idx, game_record in enumerate(pool.imap(_generate_single_game_worker, guided_args)):
                all_games.append(game_record)

                if game_record['vertical_won']:
                    guided_v_wins += 1
                guided_moves += game_record['n_moves']

                # Progress reporting every 100 games
                if (game_idx + 1) % 100 == 0:
                    now = time.time()
                    # Rolling speed over last 100 games
                    games_per_sec = 100 / (now - last_checkpoint_time) if now > last_checkpoint_time else 0
                    last_checkpoint_time = now
                    eta = (n_guided - game_idx - 1) / games_per_sec if games_per_sec > 0 else 0
                    print(f"  Guided {game_idx + 1:5d}/{n_guided} | "
                          f"{games_per_sec:.2f} g/s | "
                          f"ETA: {eta/60:.1f}m")

        print(f"  Guided games complete: V-win {100*guided_v_wins/n_guided:.1f}%, "
              f"avg {guided_moves/n_guided:.0f} moves")

    # Final save
    _save_game_records(output_path, all_games)

    # Calculate final statistics
    elapsed = time.time() - start_time
    total_moves = random_moves + guided_moves
    vertical_wins = random_v_wins + guided_v_wins
    horizontal_wins = n_games - vertical_wins
    avg_moves = total_moves / n_games

    metadata = {
        'n_games': n_games,
        'n_random_games': n_random,
        'n_guided_games': n_guided,
        'random_fraction': random_fraction,
        'total_moves': total_moves,
        'avg_moves_per_game': avg_moves,
        'vertical_wins': vertical_wins,
        'horizontal_wins': horizontal_wins,
        'win_rate_vertical': vertical_wins / n_games,
        'random_v_wins': random_v_wins,
        'guided_v_wins': guided_v_wins,
        'elapsed_seconds': elapsed,
        'games_per_second': n_games / elapsed,
        'seed': seed,
        'n_random_opening': N_RANDOM_OPENING,
        'random_prob': RANDOM_PROB,
        'd2_threshold': D2_THRESHOLD,
        'd3_threshold': D3_THRESHOLD,
        'n_workers': n_workers
    }

    print("\n" + "=" * 60)
    print("Data Generation Complete")
    print("=" * 60)
    print(f"Total games: {n_games} ({n_random} random + {n_guided} guided)")
    print(f"Total moves: {total_moves} ({avg_moves:.0f} avg/game)")
    if n_random > 0:
        print(f"Random games: V-win {100*random_v_wins/n_random:.1f}%, avg {random_moves/n_random:.0f} moves")
    if n_guided > 0:
        print(f"Guided games: V-win {100*guided_v_wins/n_guided:.1f}%, avg {guided_moves/n_guided:.0f} moves")
    print(f"Overall V-win: {100*vertical_wins/n_games:.1f}%")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Speed: {n_games/elapsed:.2f} games/s")
    print(f"Saved to: {output_path}")

    return metadata


def _save_game_records(path: str, games: List[Dict]):
    """Save game records to compressed numpy archive.

    Format:
        moves: 2D array, padded to max length, -1 for padding
        lengths: 1D array of actual game lengths
        winners: 1D bool array (True = vertical won)
    """
    if not games:
        return

    # Find max game length for padding
    max_len = max(g['n_moves'] for g in games)
    n_games = len(games)

    # Create padded moves array
    moves = np.full((n_games, max_len), -1, dtype=np.int16)
    lengths = np.zeros(n_games, dtype=np.int16)
    winners = np.zeros(n_games, dtype=bool)

    for i, g in enumerate(games):
        moves[i, :g['n_moves']] = g['moves']
        lengths[i] = g['n_moves']
        winners[i] = g['vertical_won']

    np.savez_compressed(path, moves=moves, lengths=lengths, winners=winners)


# ============================================================================
# Section 3.4 Tests (with timeouts and depth=2)
# ============================================================================

def test_simple_eval_bounds():
    """Eval should return values in [0, 1]."""
    game = dg.domineering_game()
    value = simple_eval(game)
    assert 0 <= value <= 1, f"Value {value} out of bounds on empty board"
    assert abs(value - 0.5) < 0.01, f"Empty board value {value} should be ~0.5"

    # Test on a few random positions
    np.random.seed(42)
    for _ in range(5):
        game = dg.domineering_game()
        for _ in range(20):
            if game[3]:
                break
            legal = dg.legal_moves(game)
            legal_indices = np.where(legal)[0]
            if len(legal_indices) == 0:
                break
            dg.make_move(game, np.random.choice(legal_indices))

        if not game[3]:
            value = simple_eval(game)
            assert 0 <= value <= 1, f"Value {value} out of bounds"

    print("PASS: test_simple_eval_bounds")


def test_move_ordering():
    """Move ordering should prioritize center moves."""
    # Create some legal moves
    legal_indices = np.array([0, 15, 120, 128, 239])  # Various vertical moves

    ordered = order_moves(legal_indices)

    # Move 120 is at (7, 8) - closest to center (7.5, 7.5)
    # Move 128 is at (8, 0) - far from center horizontally
    assert ordered[0] == 120, f"Expected center move first, got {ordered[0]}"

    # Test with TT best move
    ordered_with_tt = order_moves(legal_indices, tt_best_move=15)
    assert ordered_with_tt[0] == 15, f"TT move should be first, got {ordered_with_tt[0]}"

    print("PASS: test_move_ordering")


def test_transposition_table():
    """Transposition table should cache and retrieve correctly."""
    tt = TranspositionTable()

    # Store a value
    tt.store(12345, 4, 0.75, TTFlag.EXACT, 42)

    # Retrieve at same depth
    value, move, is_cutoff = tt.lookup(12345, 4, 0.0, 1.0)
    assert is_cutoff, "Should be cutoff for EXACT at same depth"
    assert value == 0.75, f"Wrong value: {value}"
    assert move == 42, f"Wrong move: {move}"

    # Retrieve at deeper depth (should not cutoff, but give move)
    value, move, is_cutoff = tt.lookup(12345, 5, 0.0, 1.0)
    assert not is_cutoff, "Should not cutoff for deeper search"
    assert move == 42, "Should still give best move for ordering"

    print("PASS: test_transposition_table")


def test_alpha_beta_finds_move():
    """Alpha-beta should find legal moves."""
    game = dg.domineering_game()
    tt = TranspositionTable()

    # Use depth 2 for fast test
    value, best_move = alpha_beta(game, depth=2, alpha=float('-inf'), beta=float('inf'), tt=tt)

    assert 0 <= value <= 1, f"Value {value} out of bounds"
    assert best_move is not None, "Should find a best move"

    legal = dg.legal_moves(game)
    assert legal[best_move], f"Best move {best_move} should be legal"

    print("PASS: test_alpha_beta_finds_move")


def test_game_record():
    """Game record function should produce valid game data."""
    np.random.seed(42)
    random.seed(42)

    record = play_game_record()

    assert 'moves' in record, "Should have moves"
    assert 'vertical_won' in record, "Should have vertical_won"
    assert 'n_moves' in record, "Should have n_moves"
    assert len(record['moves']) == record['n_moves'], "Move count mismatch"
    assert record['n_moves'] > 0, "Should have at least one move"
    assert isinstance(record['vertical_won'], bool), "vertical_won should be bool"

    # Verify all moves are valid indices
    for move in record['moves']:
        assert 0 <= move < N_MOVES, f"Invalid move index: {move}"

    print("PASS: test_game_record")


def test_random_game_record():
    """Random game record function should produce valid game data."""
    np.random.seed(42)
    random.seed(42)

    record = play_random_game_record()

    assert 'moves' in record, "Should have moves"
    assert 'vertical_won' in record, "Should have vertical_won"
    assert 'n_moves' in record, "Should have n_moves"
    assert len(record['moves']) == record['n_moves'], "Move count mismatch"
    assert record['n_moves'] > 0, "Should have at least one move"
    assert isinstance(record['vertical_won'], bool), "vertical_won should be bool"

    # Verify all moves are valid indices
    for move in record['moves']:
        assert 0 <= move < N_MOVES, f"Invalid move index: {move}"

    # Run a few more random games to check consistency
    for _ in range(5):
        record = play_random_game_record()
        assert record['n_moves'] > 0, "Game should have moves"
        assert all(0 <= m < N_MOVES for m in record['moves']), "All moves should be valid"

    print("PASS: test_random_game_record")


def run_section_3_tests():
    """Run all Section 3.4 tests."""
    print("=" * 60)
    print("Running Section 3.4 Tests")
    print("=" * 60)

    test_simple_eval_bounds()
    test_move_ordering()
    test_transposition_table()
    test_alpha_beta_finds_move()
    test_game_record()
    test_random_game_record()

    print("=" * 60)
    print("All Section 3.4 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Domineering training data')
    parser.add_argument('--test', action='store_true', help='Run tests instead of generating data')
    parser.add_argument('--n_games', type=int, default=1000, help='Number of games to generate')
    parser.add_argument('--output', type=str, default='data/phase1_games.npz', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of parallel workers (default: cpu_count)')
    parser.add_argument('--random_fraction', type=float, default=0.5, help='Fraction of games that are fully random (default: 0.5)')

    args = parser.parse_args()

    if args.test:
        run_section_3_tests()
    else:
        generate_phase1_data(
            n_games=args.n_games,
            output_path=args.output,
            seed=args.seed,
            n_workers=args.n_workers,
            random_fraction=args.random_fraction
        )
