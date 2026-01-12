# Domineering Embedding Experiment

Embed a small pretrained transformer into a larger one and see if the representation persists. The tagline in my head is "spoonful of sugar"—we've rigged the lottery by giving the large network circuits that are already well-adapted to a subtask, and for combinatorial reasons the winning tickets are likely to make use of them.

## The Setup

**Domain**: Domineering on a 16x16 board.

**Auxiliary task**: Divide the board into a 4x4 grid of 4x4 sectors. For each sector, predict the difference between legal vertical moves and legal horizontal moves in that sector. This is plausibly useful (local analysis matters in combinatorial games), cheap to compute, and encourages a particular internal representation we can probe for.

**Architecture**: Both networks are transformers with latent dimension 128 and head dimension 16. Small network: 3 layers, 4 heads, 512 MLP neurons. Large network: 6 layers, 8 heads, 768 MLP neurons. We store attention weights head-by-head rather than concatenating them, to make the embedding math easier.

**The embedding**: Copy the small network into layers 0-2 of the large network. The tricky part is fresh components (new heads and MLP neurons). Solution: **set their output weights to zero**. For fresh attention heads, zero W_O. For fresh MLP neurons, zero the corresponding columns of W_out. The residual stream through embedded layers is now exactly the same as in the small network. But gradients still flow through the zeros, so fresh components can wake up during training.

**What we're testing**: Large models are never trained with sector loss. We probe to see whether the representation persists from the embedding. The key question: at layer 2 (the last embedded layer), does the sector representation persist in Large+embed(aux) even without direct supervision?

## Replication

### Requirements

```
numpy torch matplotlib scikit-learn tqdm
```

### Generate data

```bash
python data_gen.py --n_games 25000 --output data/games.npz --workers 8
```

Takes 6-10 hours on 8 cores. Data is self-play from a weak model with 1-ply lookahead—not high quality, just what I could generate for the board size I wanted with the compute I was willing to spend.

### Run experiment

```bash
python run_experiment.py
```

Trains 5 model types across 3 seeds:
- Small+aux (trained with sector loss)
- Small-noaux (trained without sector loss)
- Large-baseline (random init)
- Large+embed(aux) (small+aux embedded)
- Large+embed(noaux) (small-noaux embedded)

Outputs checkpoints to `checkpoints/`, training histories to `results/histories.json`, plots to `plots/`.

### Analyze

```bash
python analyze_asymptotic_gap.py --full
```

Fits decay curves to the probe R^2 gap between conditions, bootstraps confidence intervals, runs some sanity checks. Results go to `analysis_results/`.

## Files

| File | Purpose |
|------|---------|
| `domineering_game.py` | Game logic, move representation |
| `model.py` | Transformer architecture |
| `config.py` | Hyperparameters |
| `data_gen.py` | Alpha-beta game generation |
| `data_loader.py` | Efficient batching with on-the-fly augmentation |
| `training.py` | Training loop |
| `embedding.py` | Small-to-large embedding + verification |
| `probing.py` | Linear probes |
| `run_experiment.py` | Main experiment runner |
| `analyze_asymptotic_gap.py` | Post-experiment analysis |

## Note on code quality

Most of this code wasn't written by hand. I don't think there are major errors that I missed on visual inspection, but it's possible, since I did this on a tight timeline. Nothing here is particularly complicated and I described the intricate steps in enough detail that I assume they were done correctly, but put slightly less confidence into experiment fidelity than you would if AI coding tools didn't exist.
