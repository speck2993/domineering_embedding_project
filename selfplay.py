"""Iterative self-play data generation with 1-ply lookahead.

Generates high-quality training data through iterative improvement:
1. Generate games via deterministic 1-ply lookahead (random opening)
2. Train model on accumulated games
3. Compare new vs old model with duplicate games
4. Keep winner, accumulate data until improvement

Features:
- Parallel game generation for GPU efficiency
- Persistent block-based storage for crash resilience
- Clean progress reporting
"""

import argparse
import copy
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

import domineering_game as dg
from domineering_game import copy_game
from model import create_small_model, create_medium_model, create_large_model, game_to_tokens, count_parameters
from data_loader import EfficientDomineeringDataset
from training import train_model, collate_batch


# Constants
N_RANDOM_OPENING = 16  # Random moves at start of each game
SELFPLAY_DIR = 'data/selfplay'


# ============================================================================
# Console Output Helpers
# ============================================================================

def print_progress(msg, end='\n'):
    """Print with line clearing for clean updates."""
    sys.stdout.write(f"\r\033[K{msg}")
    if end:
        sys.stdout.write(end)
    sys.stdout.flush()


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# ============================================================================
# Value Network Wrapper
# ============================================================================

class ValuePredictor:
    """Wrapper for value network inference with batching."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    @torch.inference_mode()
    def predict_batch(self, games):
        """Predict values for multiple game states.

        Args:
            games: List of game states

        Returns:
            numpy array of values (from vertical's perspective)
        """
        if not games:
            return np.array([])

        tokens_list = [game_to_tokens(g) for g in games]
        tokens = torch.from_numpy(np.stack(tokens_list)).to(self.device)

        masks = torch.stack([
            torch.from_numpy(dg.legal_moves(g).copy())
            for g in games
        ]).to(self.device)

        value, _, _ = self.model(tokens, masks)
        return torch.sigmoid(value.squeeze(-1)).cpu().numpy()


# ============================================================================
# Parallel Game Runner
# ============================================================================

class ParallelGameRunner:
    """Runs multiple games in parallel with batched neural network evaluation."""

    def __init__(self, predictor, n_parallel=16, n_random_opening=N_RANDOM_OPENING):
        self.predictor = predictor
        self.n_parallel = n_parallel
        self.n_random_opening = n_random_opening

    def generate_games(self, n_games, seed=42, progress_interval=60):
        """Generate n_games using parallel execution.

        Args:
            n_games: Total games to generate
            seed: Base random seed
            progress_interval: Seconds between progress updates

        Returns:
            List of game records
        """
        completed_games = []
        active_games = []
        active_moves = []
        active_seeds = []
        next_seed = seed

        start_time = time.time()
        last_report = start_time
        v_wins = 0
        total_moves = 0

        # Initialize first batch of games
        for _ in range(min(self.n_parallel, n_games)):
            game = dg.domineering_game()
            active_games.append(game)
            active_moves.append([])
            active_seeds.append(next_seed)
            np.random.seed(next_seed)
            next_seed += 1

        while active_games:
            # Collect all games that need NN evaluation
            games_needing_eval = []
            children_per_game = []
            legal_indices_per_game = []

            for i, game in enumerate(active_games):
                legal = dg.legal_moves(game)
                legal_indices = np.where(legal)[0]

                if len(legal_indices) == 0:
                    # Game should be done
                    children_per_game.append([])
                    legal_indices_per_game.append(legal_indices)
                    continue

                # Random opening phase
                if len(active_moves[i]) < self.n_random_opening:
                    np.random.seed(active_seeds[i] + len(active_moves[i]))
                    chosen_move = int(np.random.choice(legal_indices))
                    active_moves[i].append(chosen_move)
                    dg.make_move(game, chosen_move)
                    children_per_game.append([])
                    legal_indices_per_game.append(legal_indices)
                else:
                    # Need NN evaluation - create children
                    children = []
                    for move in legal_indices:
                        child = copy_game(game)
                        dg.make_move(child, move)
                        children.append(child)
                    children_per_game.append(children)
                    legal_indices_per_game.append(legal_indices)
                    games_needing_eval.extend(children)

            # Batch evaluate all children
            if games_needing_eval:
                all_values = self.predictor.predict_batch(games_needing_eval)

                # Distribute values back and make moves
                value_idx = 0
                for i, (game, children, legal_indices) in enumerate(
                    zip(active_games, children_per_game, legal_indices_per_game)
                ):
                    if not children:  # Was random move or no legal moves
                        continue

                    n_children = len(children)
                    child_values = all_values[value_idx:value_idx + n_children]
                    value_idx += n_children

                    # Pick best move
                    if game[1]:  # Horizontal wants LOW
                        best_idx = np.argmin(child_values)
                    else:  # Vertical wants HIGH
                        best_idx = np.argmax(child_values)

                    chosen_move = int(legal_indices[best_idx])
                    active_moves[i].append(chosen_move)
                    dg.make_move(game, chosen_move)

            # Check for completed games
            i = 0
            while i < len(active_games):
                game = active_games[i]
                if game[3]:  # Game is done
                    vertical_won = game[1]
                    record = {
                        'moves': active_moves[i],
                        'vertical_won': vertical_won,
                        'n_moves': len(active_moves[i])
                    }
                    completed_games.append(record)

                    if vertical_won:
                        v_wins += 1
                    total_moves += record['n_moves']

                    # Remove completed game
                    active_games.pop(i)
                    active_moves.pop(i)
                    active_seeds.pop(i)

                    # Start new game if needed
                    if len(completed_games) + len(active_games) < n_games:
                        new_game = dg.domineering_game()
                        active_games.append(new_game)
                        active_moves.append([])
                        active_seeds.append(next_seed)
                        np.random.seed(next_seed)
                        next_seed += 1
                else:
                    i += 1

            # Progress reporting
            now = time.time()
            if now - last_report >= progress_interval:
                elapsed = now - start_time
                n_done = len(completed_games)
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = n_games - n_done
                eta = remaining / rate if rate > 0 else 0
                v_pct = 100 * v_wins / n_done if n_done > 0 else 0
                avg_len = total_moves / n_done if n_done > 0 else 0

                print_progress(
                    f"  Games: {n_done}/{n_games} | {rate:.2f}/sec | "
                    f"V-win: {v_pct:.0f}% | Avg len: {avg_len:.0f} | ETA: {format_duration(eta)}",
                    end=''
                )
                last_report = now

        # Final newline
        if len(completed_games) > 0:
            print_progress(
                f"  Games: {len(completed_games)}/{n_games} complete | "
                f"V-win: {100*v_wins/len(completed_games):.0f}% | "
                f"Avg len: {total_moves/len(completed_games):.0f}"
            )

        return completed_games


# ============================================================================
# Game Generation Utilities
# ============================================================================

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
            break
        moves.append(move)
        dg.make_move(game, move)

    # Continue with 1-ply lookahead
    while not game[3]:
        legal = dg.legal_moves(game)
        legal_indices = np.where(legal)[0]

        if len(legal_indices) == 0:
            break

        predictor = predictor_h if game[1] else predictor_v

        children = []
        for move in legal_indices:
            child = copy_game(game)
            dg.make_move(child, move)
            children.append(child)

        child_values = predictor.predict_batch(children)

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


def generate_random_opening(seed, n_moves=10):
    """Generate a random opening sequence."""
    np.random.seed(seed)
    game = dg.domineering_game()
    moves = []

    for _ in range(n_moves):
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

def mcnemar_test(decisive_a, decisive_b):
    """Compute McNemar's test for paired binary outcomes.

    Args:
        decisive_a: Number of openings where A won both games
        decisive_b: Number of openings where B won both games

    Returns:
        p_value: Two-sided p-value from McNemar's test
    """
    # McNemar's test: compare discordant pairs
    # Under null hypothesis, b and c should be equal
    b, c = decisive_a, decisive_b

    if b + c == 0:
        return 1.0  # No decisive outcomes, no evidence either way

    # Use normal approximation (valid when b + c >= 25)
    # For smaller samples, could use exact binomial test
    import math
    z = (b - c) / math.sqrt(b + c)

    # Two-sided p-value from standard normal
    # P(|Z| > |z|) = 2 * P(Z > |z|)
    from math import erf
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))

    return p_value


def compare_models(model_a, model_b, n_openings=150, device='cpu'):
    """Compare two models with duplicate games for fairness.

    For each opening, plays two games (swapping colors). Outcomes:
    - Decisive for A: A wins both games
    - Decisive for B: B wins both games
    - Wash: Each model wins one game

    Uses McNemar's test on decisive pairs to determine significance.
    """
    pred_a = ValuePredictor(model_a, device)
    pred_b = ValuePredictor(model_b, device)

    decisive_a = 0  # Openings where A won both
    decisive_b = 0  # Openings where B won both
    washes = 0      # Openings where each won one
    games = []

    for i in range(n_openings):
        opening = generate_random_opening(seed=i * 12345, n_moves=10)

        # Game 1: A = vertical, B = horizontal
        result1 = play_game_from_opening(pred_a, pred_b, opening)
        games.append(result1)
        a_won_as_v = result1['vertical_won']

        # Game 2: A = horizontal, B = vertical
        result2 = play_game_from_opening(pred_b, pred_a, opening)
        games.append(result2)
        a_won_as_h = not result2['vertical_won']

        # Classify this opening
        if a_won_as_v and a_won_as_h:
            decisive_a += 1
        elif not a_won_as_v and not a_won_as_h:
            decisive_b += 1
        else:
            washes += 1

    # Compute McNemar's test
    p_value = mcnemar_test(decisive_a, decisive_b)

    # A is significantly better if it has more decisive wins AND p < 0.05
    a_significantly_better = decisive_a > decisive_b and p_value < 0.05

    # Also compute traditional win counts for backwards compatibility
    a_wins = decisive_a * 2 + washes
    b_wins = decisive_b * 2 + washes

    return {
        'decisive_a': decisive_a,
        'decisive_b': decisive_b,
        'washes': washes,
        'p_value': p_value,
        'a_significantly_better': a_significantly_better,
        # Legacy fields
        'a_wins': a_wins,
        'b_wins': b_wins,
        'total': 2 * n_openings,
        'a_win_rate': a_wins / (2 * n_openings),
        'games': games
    }


# ============================================================================
# Persistent Storage
# ============================================================================

def save_block(games, block_num, directory=SELFPLAY_DIR):
    """Save a block of games to persistent storage."""
    os.makedirs(directory, exist_ok=True)

    n_games = len(games)
    max_len = max(g['n_moves'] for g in games)

    moves_array = np.full((n_games, max_len), -1, dtype=np.int16)
    lengths = np.zeros(n_games, dtype=np.int16)
    winners = np.zeros(n_games, dtype=bool)

    for i, g in enumerate(games):
        moves_array[i, :g['n_moves']] = g['moves']
        lengths[i] = g['n_moves']
        winners[i] = g['vertical_won']

    block_path = os.path.join(directory, f'block_{block_num:04d}.npz')
    np.savez_compressed(block_path, moves=moves_array, lengths=lengths, winners=winners)

    # Update manifest
    manifest_path = os.path.join(directory, 'manifest.txt')
    with open(manifest_path, 'a') as f:
        f.write(f'block_{block_num:04d}.npz\n')

    return {
        'path': block_path,
        'n_games': n_games,
        'total_positions': int(lengths.sum()),
        'avg_length': float(lengths.mean())
    }


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
                       n_parallel=16, device='cuda', seed=42):
    """Main iterative self-play loop.

    Args:
        initial_model_path: Path to initial trained model checkpoint
        target_games: Total number of games to generate
        output_path: Path to save final NPZ file
        batch_size: Games per iteration
        comparison_openings: Openings for model comparison (total games = 2x)
        n_parallel: Number of parallel games for generation
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
    baseline_model = create_medium_model()
    checkpoint = torch.load(initial_model_path, map_location=device, weights_only=True)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    baseline_model.to(device)
    baseline_model.eval()

    print(f"Model parameters: {count_parameters(baseline_model):,}")

    # State
    all_games = []
    pending_games = []
    total_generated = 0
    model_updates = 0
    iteration = 0
    block_num = 0

    # Clear/create selfplay directory
    os.makedirs(SELFPLAY_DIR, exist_ok=True)
    manifest_path = os.path.join(SELFPLAY_DIR, 'manifest.txt')
    open(manifest_path, 'w').close()  # Clear manifest

    start_time = time.time()

    while total_generated < target_games:
        iteration += 1
        iter_start = time.time()

        # Progress header
        elapsed = iter_start - start_time
        if total_generated > 0:
            rate = total_generated / (elapsed / 60)
            remaining = target_games - total_generated
            eta = (remaining / rate) * 60
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} | {total_generated}/{target_games} ({100*total_generated/target_games:.0f}%) | "
                  f"{rate:.1f}/min | ETA: {format_duration(eta)}")
        else:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} | Starting...")
        print("=" * 60)

        # Generate games with parallel runner
        print(f"\nGenerating {batch_size} games (parallel={n_parallel})...")
        gen_start = time.time()

        predictor = ValuePredictor(baseline_model, device)
        runner = ParallelGameRunner(predictor, n_parallel=n_parallel)
        new_games = runner.generate_games(batch_size, seed=seed + total_generated)

        pending_games.extend(new_games)
        total_generated += batch_size
        gen_elapsed = time.time() - gen_start
        print(f"  Generated in {format_duration(gen_elapsed)} ({batch_size/gen_elapsed:.1f}/sec)")

        # Save block to disk (crash resilience)
        block_num += 1
        block_stats = save_block(new_games, block_num)
        print(f"  Saved {block_stats['path']}")

        # Train new model
        train_start = time.time()
        print(f"\nTraining on {len(pending_games)} pending games...")

        # Save pending games for training
        temp_path = os.path.join(SELFPLAY_DIR, 'temp_training.npz')
        games_to_npz(pending_games, temp_path)

        new_model = create_medium_model()
        new_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))
        new_model.to(device)

        train_dataset = EfficientDomineeringDataset(temp_path, split='train')
        val_dataset = EfficientDomineeringDataset(temp_path, split='val')

        # Precompute positions
        train_dataset.precompute_epoch()
        val_dataset.precompute_epoch()

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False,  # Already shuffled internally
                                  num_workers=0, collate_fn=collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                                num_workers=0, collate_fn=collate_batch)

        train_model(new_model, train_loader, val_loader, n_epochs=1,
                    use_auxiliary=False, value_only=True, device=device)
        train_elapsed = time.time() - train_start
        print(f"  Training completed in {format_duration(train_elapsed)}")

        # Compare models
        compare_start = time.time()
        print(f"\nComparing models ({comparison_openings * 2} games)...")
        comparison = compare_models(new_model, baseline_model,
                                    n_openings=comparison_openings, device=device)
        compare_elapsed = time.time() - compare_start

        pending_games.extend(comparison['games'])

        print(f"  Decisive: new={comparison['decisive_a']}, old={comparison['decisive_b']}, washes={comparison['washes']}")
        print(f"  McNemar p-value: {comparison['p_value']:.4f}")
        print(f"  Comparison completed in {format_duration(compare_elapsed)}")

        # Decide whether to keep new model (must be significantly better at p < 0.05)
        if comparison['a_significantly_better']:
            print("  -> New model significantly better (p < 0.05)! Updating baseline.")
            baseline_model = new_model
            all_games.extend(pending_games)
            pending_games = []
            model_updates += 1
        else:
            print("  -> No significant improvement. Accumulating more data.")

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        iter_elapsed = time.time() - iter_start
        print(f"\nIteration {iteration} completed in {format_duration(iter_elapsed)}")

    # Final save
    all_games.extend(pending_games)

    print(f"\n{'='*60}")
    print("Saving final dataset...")
    stats = games_to_npz(all_games, output_path)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {format_duration(elapsed)}")
    print(f"  Total games: {stats['n_games']}")
    print(f"  Total positions: {stats['total_positions']:,}")
    print(f"  Model updates: {model_updates}")
    print(f"  Blocks saved: {block_num}")
    print(f"  Output: {output_path}")

    return {
        **stats,
        'model_updates': model_updates,
        'elapsed': elapsed
    }


# ============================================================================
# Simple Generation (No Iterative Training)
# ============================================================================

def generate_with_trained_model(model_path, n_games, output_path,
                                 n_parallel=16, device='cuda', seed=42,
                                 model_type='large'):
    """Generate games using a pre-trained model (no iterative improvement).

    This is faster than iterative_selfplay() because it doesn't retrain the model.
    Use this to generate additional training data using an already-trained model.

    Args:
        model_path: Path to trained model checkpoint (.pt file)
        n_games: Number of games to generate
        output_path: Where to save the NPZ file
        n_parallel: Number of parallel games during generation
        device: Device for inference
        seed: Random seed
        model_type: 'large', 'small', or 'medium' (default 'large')

    Returns:
        Dict with statistics
    """
    print("=" * 60)
    print("Self-Play Generation (Fixed Model)")
    print("=" * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model
    print(f"\nLoading model from {model_path}")
    if model_type == 'large':
        model = create_large_model()
    elif model_type == 'small':
        model = create_small_model()
    else:
        model = create_medium_model()

    # Handle both full checkpoint and state_dict only
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"Model parameters: {count_parameters(model):,}")

    # Generate games
    print(f"\nGenerating {n_games} games (parallel={n_parallel})...")
    start_time = time.time()

    predictor = ValuePredictor(model, device)
    runner = ParallelGameRunner(predictor, n_parallel=n_parallel)
    games = runner.generate_games(n_games, seed=seed, progress_interval=30)

    gen_elapsed = time.time() - start_time
    print(f"\nGenerated {len(games)} games in {format_duration(gen_elapsed)} ({len(games)/gen_elapsed:.1f}/sec)")

    # Save to NPZ
    stats = games_to_npz(games, output_path)
    print(f"Saved to {output_path}")
    print(f"  Total games: {stats['n_games']}")
    print(f"  Total positions: {stats['total_positions']:,}")
    print(f"  Avg game length: {stats['avg_length']:.1f}")

    # Win rate analysis
    v_wins = sum(1 for g in games if g['vertical_won'])
    print(f"  Vertical win rate: {100*v_wins/len(games):.1f}%")

    return {**stats, 'elapsed': gen_elapsed}


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Self-play data generation')
    subparsers = parser.add_subparsers(dest='mode', help='Generation mode')

    # Iterative self-play (original)
    iter_parser = subparsers.add_parser('iterative', help='Iterative self-play with model improvement')
    iter_parser.add_argument('--initial_model', type=str, required=True,
                            help='Path to initial trained model checkpoint')
    iter_parser.add_argument('--target_games', type=int, default=50000,
                            help='Total number of games to generate')
    iter_parser.add_argument('--output', type=str, default='data/selfplay_games.npz',
                            help='Output file path')
    iter_parser.add_argument('--batch_size', type=int, default=2500,
                            help='Games per iteration')
    iter_parser.add_argument('--comparison_openings', type=int, default=150,
                            help='Openings for model comparison (total games = 2x this)')
    iter_parser.add_argument('--n_parallel', type=int, default=16,
                            help='Number of parallel games during generation')
    iter_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')

    # Simple generation (new)
    gen_parser = subparsers.add_parser('generate', help='Generate games with fixed trained model')
    gen_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model checkpoint')
    gen_parser.add_argument('--n_games', type=int, default=50000,
                           help='Number of games to generate')
    gen_parser.add_argument('--output', type=str, default='data/selfplay_additional.npz',
                           help='Output file path')
    gen_parser.add_argument('--n_parallel', type=int, default=16,
                           help='Number of parallel games during generation')
    gen_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed')
    gen_parser.add_argument('--model_type', type=str, default='large',
                           choices=['large', 'small', 'medium'],
                           help='Type of model architecture')

    # Test mode
    test_parser = subparsers.add_parser('test', help='Run self-play tests')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.mode == 'iterative':
        iterative_selfplay(
            initial_model_path=args.initial_model,
            target_games=args.target_games,
            output_path=args.output,
            batch_size=args.batch_size,
            comparison_openings=args.comparison_openings,
            n_parallel=args.n_parallel,
            device=device,
            seed=args.seed
        )
    elif args.mode == 'generate':
        generate_with_trained_model(
            model_path=args.model,
            n_games=args.n_games,
            output_path=args.output,
            n_parallel=args.n_parallel,
            device=device,
            seed=args.seed,
            model_type=args.model_type
        )
    elif args.mode == 'test':
        run_selfplay_tests()
    else:
        # Default: show help
        parser.print_help()


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


def test_parallel_game_runner():
    """Test parallel game generation."""
    model = create_small_model()
    predictor = ValuePredictor(model, device='cpu')

    runner = ParallelGameRunner(predictor, n_parallel=4, n_random_opening=10)
    games = runner.generate_games(8, seed=42, progress_interval=9999)

    assert len(games) == 8, f"Expected 8 games, got {len(games)}"

    for g in games:
        assert 'moves' in g
        assert 'vertical_won' in g
        assert 'n_moves' in g
        assert len(g['moves']) == g['n_moves']
        assert g['n_moves'] > 10

    print("PASS: test_parallel_game_runner")


def test_generate_random_opening():
    """Test random opening generation."""
    opening1 = generate_random_opening(seed=42, n_moves=N_RANDOM_OPENING)
    opening2 = generate_random_opening(seed=42, n_moves=N_RANDOM_OPENING)
    opening3 = generate_random_opening(seed=43, n_moves=N_RANDOM_OPENING)

    assert opening1 == opening2, "Same seed should give same opening"
    assert opening1 != opening3, "Different seeds should give different openings"
    assert len(opening1) == N_RANDOM_OPENING

    print("PASS: test_generate_random_opening")


def test_compare_models():
    """Test model comparison with duplicate games."""
    model_a = create_small_model()
    model_b = create_small_model()

    result = compare_models(model_a, model_b, n_openings=10, device='cpu')

    assert 'a_wins' in result
    assert 'b_wins' in result
    assert result['a_wins'] + result['b_wins'] == result['total']
    assert result['total'] == 20

    print(f"  Model A wins: {result['a_wins']}/{result['total']}")
    print("PASS: test_compare_models")


def run_selfplay_tests():
    """Run all self-play tests."""
    print("=" * 60)
    print("Running Self-Play Tests")
    print("=" * 60)

    np.random.seed(42)
    test_value_predictor()
    test_parallel_game_runner()
    test_generate_random_opening()
    test_compare_models()

    print("=" * 60)
    print("All self-play tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
