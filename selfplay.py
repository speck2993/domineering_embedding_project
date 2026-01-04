"""Iterative self-play data generation with 1-ply lookahead.

Generates high-quality training data through iterative improvement:
1. Generate games via deterministic 1-ply lookahead (random opening)
2. Train model on accumulated games
3. Compare new vs old model with duplicate games
4. Keep winner, accumulate data until improvement
"""

import argparse
import copy
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

import domineering_game as dg
from model import create_small_model, create_large_model, game_to_tokens, count_parameters
from data_loader import DomineeringDataset
from training import train_model, collate_batch


# Constants
N_RANDOM_OPENING = 10  # Random moves at start of each game


# ============================================================================
# Value Network Wrapper
# ============================================================================

class ValuePredictor:
    """Wrapper for value network inference."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def predict_batch(self, games):
        """Predict values for multiple game states.

        Args:
            games: List of game states

        Returns:
            numpy array of values (from vertical's perspective)
        """
        tokens_list = [game_to_tokens(g) for g in games]
        tokens = torch.from_numpy(np.stack(tokens_list)).to(self.device)

        masks = torch.stack([
            torch.from_numpy(dg.legal_moves(g).copy())
            for g in games
        ]).to(self.device)

        value, _, _ = self.model(tokens, masks)
        return value.squeeze(-1).cpu().numpy()


# ============================================================================
# Game Generation
# ============================================================================

def play_game_deterministic(predictor, n_random_opening=N_RANDOM_OPENING, seed=None):
    """Play a game: random opening, then deterministic 1-ply lookahead.

    Args:
        predictor: ValuePredictor for evaluating positions
        n_random_opening: Number of random moves at start
        seed: Random seed for opening moves

    Returns:
        Dict with 'moves', 'vertical_won', 'n_moves'
    """
    if seed is not None:
        np.random.seed(seed)

    game = dg.domineering_game()
    moves = []

    while not game[3]:  # While game not done
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]

        if len(legal_indices) == 0:
            break

        # Random opening phase
        if len(moves) < n_random_opening:
            chosen_move = int(np.random.choice(legal_indices))
        else:
            # Deterministic 1-ply lookahead
            children = []
            for move in legal_indices:
                child = copy.deepcopy(game)
                dg.make_move(child, move)
                children.append(child)

            child_values = predictor.predict_batch(children)

            # Current player picks best move
            # Vertical (game[1]=False) wants HIGH value
            # Horizontal (game[1]=True) wants LOW value
            if game[1]:  # Horizontal's turn
                best_idx = np.argmin(child_values)
            else:  # Vertical's turn
                best_idx = np.argmax(child_values)

            chosen_move = int(legal_indices[best_idx])

        moves.append(chosen_move)
        dg.make_move(game, chosen_move)

    # Winner: if game[1] is True (horizontal's turn) and game is done, vertical won
    vertical_won = game[1]

    return {
        'moves': moves,
        'vertical_won': vertical_won,
        'n_moves': len(moves)
    }


def play_game_from_opening(predictor_v, predictor_h, opening_moves):
    """Play a game from a given opening with two different predictors.

    Args:
        predictor_v: Predictor playing as vertical
        predictor_h: Predictor playing as horizontal
        opening_moves: List of moves for the opening phase

    Returns:
        Dict with 'moves', 'vertical_won', 'n_moves'
    """
    game = dg.domineering_game()
    moves = []

    # Play opening
    for move in opening_moves:
        if game[3]:
            break
        legal = dg.legal_moves(game)
        if not legal[move]:
            # Opening move not legal (shouldn't happen with proper seeds)
            break
        moves.append(move)
        dg.make_move(game, move)

    # Continue with 1-ply lookahead
    while not game[3]:
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]

        if len(legal_indices) == 0:
            break

        # Pick predictor based on current player
        predictor = predictor_h if game[1] else predictor_v

        children = []
        for move in legal_indices:
            child = copy.deepcopy(game)
            dg.make_move(child, move)
            children.append(child)

        child_values = predictor.predict_batch(children)

        # Current player picks best move
        if game[1]:  # Horizontal wants LOW
            best_idx = np.argmin(child_values)
        else:  # Vertical wants HIGH
            best_idx = np.argmax(child_values)

        chosen_move = int(legal_indices[best_idx])
        moves.append(chosen_move)
        dg.make_move(game, chosen_move)

    vertical_won = game[1]

    return {
        'moves': moves,
        'vertical_won': vertical_won,
        'n_moves': len(moves)
    }


def generate_random_opening(seed):
    """Generate a random opening sequence."""
    np.random.seed(seed)
    game = dg.domineering_game()
    moves = []

    for _ in range(N_RANDOM_OPENING):
        if game[3]:
            break
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]
        if len(legal_indices) == 0:
            break
        move = int(np.random.choice(legal_indices))
        moves.append(move)
        dg.make_move(game, move)

    return moves


# ============================================================================
# Model Comparison
# ============================================================================

def compare_models(model_a, model_b, n_openings=150, device='cpu'):
    """Compare two models with duplicate games for fairness.

    Each opening is played twice:
    - Game 1: model_a = vertical, model_b = horizontal
    - Game 2: model_a = horizontal, model_b = vertical

    Args:
        model_a: First model (the challenger/new model)
        model_b: Second model (the baseline)
        n_openings: Number of unique openings (total games = 2 * n_openings)
        device: Device for inference

    Returns:
        Dict with 'a_wins', 'b_wins', 'draws', 'a_win_rate'
    """
    pred_a = ValuePredictor(model_a, device)
    pred_b = ValuePredictor(model_b, device)

    a_wins = 0
    b_wins = 0

    for i in range(n_openings):
        opening = generate_random_opening(seed=i * 12345)

        # Game 1: A = vertical, B = horizontal
        result1 = play_game_from_opening(pred_a, pred_b, opening)
        if result1['vertical_won']:
            a_wins += 1
        else:
            b_wins += 1

        # Game 2: A = horizontal, B = vertical
        result2 = play_game_from_opening(pred_b, pred_a, opening)
        if result2['vertical_won']:
            b_wins += 1
        else:
            a_wins += 1

    total = 2 * n_openings
    return {
        'a_wins': a_wins,
        'b_wins': b_wins,
        'total': total,
        'a_win_rate': a_wins / total
    }


# ============================================================================
# Batch Game Generation
# ============================================================================

def generate_games_batch(predictor, n_games, seed=42):
    """Generate a batch of self-play games.

    Args:
        predictor: ValuePredictor to use
        n_games: Number of games to generate
        seed: Base seed

    Returns:
        List of game records
    """
    games = []
    last_report = time.time()

    for i in range(n_games):
        record = play_game_deterministic(predictor, seed=seed + i)
        games.append(record)

        if (i + 1) % 100 == 0:
            now = time.time()
            games_per_sec = 100 / (now - last_report) if now > last_report else 0
            last_report = now
            eta = (n_games - i - 1) / games_per_sec if games_per_sec > 0 else 0
            print(f"    {i+1}/{n_games} games ({games_per_sec:.1f}/sec, ETA {eta:.0f}s)")

    return games


def games_to_npz(games, output_path):
    """Convert game records to NPZ format and save."""
    n_games = len(games)
    max_len = max(g['n_moves'] for g in games)

    moves_array = np.full((n_games, max_len), -1, dtype=np.int16)
    lengths = np.zeros(n_games, dtype=np.int16)
    winners = np.zeros(n_games, dtype=bool)

    for i, g in enumerate(games):
        moves_array[i, :g['n_moves']] = g['moves']
        lengths[i] = g['n_moves']
        winners[i] = g['vertical_won']

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.savez_compressed(output_path, moves=moves_array, lengths=lengths, winners=winners)

    return {
        'n_games': n_games,
        'total_positions': int(lengths.sum()),
        'avg_length': float(lengths.mean())
    }


# ============================================================================
# Iterative Self-Play
# ============================================================================

def iterative_selfplay(initial_model_path, target_games, output_path,
                       batch_size=2500, comparison_openings=150,
                       device='cuda', seed=42):
    """Main iterative self-play loop.

    Args:
        initial_model_path: Path to initial trained model checkpoint
        target_games: Total number of games to generate
        output_path: Path to save final NPZ file
        batch_size: Games per iteration
        comparison_openings: Openings for model comparison (300 total games)
        device: Device for training and inference
        seed: Random seed

    Returns:
        Dict with statistics
    """
    print("=" * 60)
    print("Iterative Self-Play")
    print("=" * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load initial model
    print(f"\nLoading initial model from {initial_model_path}")
    baseline_model = create_large_model()
    checkpoint = torch.load(initial_model_path, map_location=device, weights_only=True)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    baseline_model.to(device)
    baseline_model.eval()

    print(f"Model parameters: {count_parameters(baseline_model):,}")

    # State
    all_games = []  # Final accumulated games
    pending_games = []  # Games since last model update
    total_generated = 0
    model_updates = 0
    iteration = 0

    start_time = time.time()

    while total_generated < target_games:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")

        # Generate batch of games
        print(f"\nGenerating {batch_size} games...")
        predictor = ValuePredictor(baseline_model, device)
        new_games = generate_games_batch(predictor, batch_size, seed=seed + total_generated)
        pending_games.extend(new_games)
        total_generated += batch_size

        print(f"  Pending games: {len(pending_games)}")
        print(f"  Total generated: {total_generated}/{target_games}")

        # Save pending games to temp file for training
        temp_path = f"temp_selfplay_{iteration}.npz"
        games_to_npz(pending_games, temp_path)

        # Train new model on pending games
        print(f"\nTraining new model on {len(pending_games)} games...")
        new_model = create_large_model()
        new_model.to(device)

        train_dataset = DomineeringDataset(temp_path, split='train')
        val_dataset = DomineeringDataset(temp_path, split='val')

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                                  num_workers=0, collate_fn=collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                                num_workers=0, collate_fn=collate_batch)

        # Train for 1 epoch, value only
        train_model(new_model, train_loader, val_loader, n_epochs=1,
                    use_auxiliary=False, value_only=True, device=device)

        # Compare models
        print(f"\nComparing new model vs baseline ({comparison_openings * 2} games)...")
        comparison = compare_models(new_model, baseline_model,
                                    n_openings=comparison_openings, device=device)

        print(f"  New model wins: {comparison['a_wins']}/{comparison['total']} "
              f"({comparison['a_win_rate']:.1%})")

        # Decide whether to keep new model
        if comparison['a_wins'] > comparison['b_wins']:
            print("  -> New model is better! Updating baseline.")
            baseline_model = new_model
            all_games.extend(pending_games)
            pending_games = []
            model_updates += 1
        else:
            print("  -> Baseline is still better. Accumulating more games.")

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Add any remaining pending games
    all_games.extend(pending_games)

    # Save final dataset
    print(f"\n{'='*60}")
    print("Saving final dataset...")
    stats = games_to_npz(all_games, output_path)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"  Total games: {stats['n_games']}")
    print(f"  Total positions: {stats['total_positions']:,}")
    print(f"  Model updates: {model_updates}")
    print(f"  Saved to {output_path}")

    return {
        **stats,
        'model_updates': model_updates,
        'elapsed': elapsed
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Iterative self-play data generation')
    parser.add_argument('--initial_model', type=str, default=None,
                        help='Path to initial trained model checkpoint')
    parser.add_argument('--target_games', type=int, default=50000,
                        help='Total number of games to generate')
    parser.add_argument('--output', type=str, default='data/selfplay_games.npz',
                        help='Output file path')
    parser.add_argument('--batch_size', type=int, default=2500,
                        help='Games per iteration')
    parser.add_argument('--comparison_openings', type=int, default=150,
                        help='Openings for model comparison (total games = 2x this)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test', action='store_true',
                        help='Run tests only')

    args = parser.parse_args()

    if args.test:
        run_selfplay_tests()
        return

    if args.initial_model is None:
        parser.error("--initial_model is required when not running tests")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    iterative_selfplay(
        initial_model_path=args.initial_model,
        target_games=args.target_games,
        output_path=args.output,
        batch_size=args.batch_size,
        comparison_openings=args.comparison_openings,
        device=device,
        seed=args.seed
    )


# ============================================================================
# Tests
# ============================================================================

def test_value_predictor():
    """Test ValuePredictor batch inference."""
    model = create_small_model()
    predictor = ValuePredictor(model, device='cpu')

    games = [dg.domineering_game() for _ in range(4)]
    for i, g in enumerate(games):
        for _ in range(i * 5):
            legal = dg.legal_moves(g)
            legal_idx = np.where(legal)[0]
            if len(legal_idx) == 0:
                break
            dg.make_move(g, np.random.choice(legal_idx))

    values = predictor.predict_batch(games)

    assert values.shape == (4,), f"Expected (4,), got {values.shape}"
    assert np.all((values >= 0) & (values <= 1)), "Values should be in [0, 1]"

    print("PASS: test_value_predictor")


def test_play_game_deterministic():
    """Test deterministic 1-ply lookahead game generation."""
    model = create_small_model()
    predictor = ValuePredictor(model, device='cpu')

    record = play_game_deterministic(predictor, n_random_opening=10, seed=42)

    assert 'moves' in record
    assert 'vertical_won' in record
    assert 'n_moves' in record
    assert len(record['moves']) == record['n_moves']
    assert record['n_moves'] > 10  # Should have more than just opening
    assert isinstance(record['vertical_won'], (bool, np.bool_))

    # Verify moves are valid
    for m in record['moves']:
        assert 0 <= m < dg.N_MOVES, f"Invalid move: {m}"

    # Verify determinism: same seed = same game
    record2 = play_game_deterministic(predictor, n_random_opening=10, seed=42)
    assert record['moves'] == record2['moves'], "Deterministic games should be identical"

    print("PASS: test_play_game_deterministic")


def test_generate_random_opening():
    """Test random opening generation."""
    opening1 = generate_random_opening(seed=42)
    opening2 = generate_random_opening(seed=42)
    opening3 = generate_random_opening(seed=43)

    assert opening1 == opening2, "Same seed should give same opening"
    assert opening1 != opening3, "Different seeds should give different openings"
    assert len(opening1) == N_RANDOM_OPENING

    print("PASS: test_generate_random_opening")


def test_compare_models():
    """Test model comparison with duplicate games."""
    model_a = create_small_model()
    model_b = create_small_model()

    # With random weights, should be roughly 50/50
    result = compare_models(model_a, model_b, n_openings=10, device='cpu')

    assert 'a_wins' in result
    assert 'b_wins' in result
    assert result['a_wins'] + result['b_wins'] == result['total']
    assert result['total'] == 20  # 10 openings * 2

    print(f"  Model A wins: {result['a_wins']}/{result['total']}")
    print("PASS: test_compare_models")


def run_selfplay_tests():
    """Run all self-play tests."""
    print("=" * 60)
    print("Running Self-Play Tests")
    print("=" * 60)

    np.random.seed(42)
    test_value_predictor()
    test_play_game_deterministic()
    test_generate_random_opening()
    test_compare_models()

    print("=" * 60)
    print("All self-play tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
