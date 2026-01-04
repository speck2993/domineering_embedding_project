# Domineering Neural Network Embedding Experiment

## Project Overview

**Goal:** Test whether embedding a small pretrained transformer into a larger transformer at initialization helps the larger network learn faster/better, by providing "known good" circuit fragments that the network can factor through.

**Core Hypothesis:** If sparse winning subnetworks are rare, and embedding provides a "known good" fragment, then winning subnetworks incorporating that fragment are vastly overrepresented among all winning subnetworks, dominating training.

**Game:** Domineering on a 16×16 board (256 squares, large enough that memorization is impossible).

**Experimental Design:** Train 20 models total:
1. 4 small networks WITH auxiliary task
2. 4 small networks WITHOUT auxiliary task
3. 4 large baseline networks (no embedding)
4. 4 large networks with small+aux embedded
5. 4 large networks with small-no-aux embedded

This tests whether benefits come from:
- The embedding itself (4 vs 3)
- The auxiliary task in the small network (4 vs 5)
- Just having more parameters (5 vs 3)

---

## Architecture Summary

| Parameter | Small | Large |
|-----------|-------|-------|
| d_model | 128 | 128 |
| d_head | 16 | 16 |
| n_heads | 4 | 8 |
| n_layers | 2 | 7 |
| d_mlp | 512 | 768 |
| ~params | 350K | 2.5M |

---

## Section 1: Core Infrastructure

### 1.1 Update Domineering for 16×16

```python
BOARD_SIZE = 16
N_MOVES = 2 * BOARD_SIZE * (BOARD_SIZE - 1)  # 480
P_MOVES = N_MOVES // 2  # 240
N_SQUARES = 256
N_SECTORS = 16  # 4×4 grid of 4×4 sectors
SECTOR_SIZE = 4
```

### 1.2 Zobrist Hashing

```python
zobrist_table = np.random.randint(0, 2**64, size=(N_SQUARES, 2), dtype=np.uint64)
zobrist_player = np.random.randint(0, 2**64, dtype=np.uint64)

def compute_hash(game):
    h = np.uint64(0)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            h ^= zobrist_table[i * BOARD_SIZE + j, game[0][i, j]]
    if game[1]:
        h ^= zobrist_player
    return h
```

### 1.3 Precompute Sector Move Masks

```python
# Build at initialization - create boolean masks for efficiency
# Each move is assigned to exactly one sector (based on top/left square)
# This simplifies debugging and interpretation
sector_v_masks = np.zeros((N_SECTORS, N_MOVES), dtype=bool)
sector_h_masks = np.zeros((N_SECTORS, N_MOVES), dtype=bool)

for move in range(N_MOVES):
    if move < P_MOVES:  # Vertical move
        i, j = divmod(move, BOARD_SIZE)
        sector_idx = (i // SECTOR_SIZE) * (BOARD_SIZE // SECTOR_SIZE) + (j // SECTOR_SIZE)
        sector_v_masks[sector_idx, move] = True
    else:  # Horizontal move
        j, i = divmod(move - P_MOVES, BOARD_SIZE)
        sector_idx = (i // SECTOR_SIZE) * (BOARD_SIZE // SECTOR_SIZE) + (j // SECTOR_SIZE)
        sector_h_masks[sector_idx, move] = True
```

### 1.4 Sector Targets (Auxiliary Task)

```python
def compute_sector_targets(game):
    """Returns (16,) array of (v_moves - h_moves) per 4×4 sector."""
    legal_moves = game[4]  # Boolean array of length 480
    targets = np.zeros(16, dtype=np.float32)
    
    for sector_idx in range(16):
        v_count = (legal_moves & sector_v_masks[sector_idx]).sum()
        h_count = (legal_moves & sector_h_masks[sector_idx]).sum()
        targets[sector_idx] = v_count - h_count
    
    return targets
```

### 1.5 Tests
- [ ] 16×16 game works
- [ ] Zobrist consistent across transpositions
- [ ] Sector targets correct on hand-crafted positions

---

## Section 2: Neural Network

### 2.1 Key Design Choice: Per-Head Weight Storage

Store W_Q, W_K, W_V, W_O as `ParameterList` of per-head matrices (not concatenated). This makes embedding trivial.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        self.W_Q = nn.ParameterList([nn.Parameter(torch.randn(d_model, d_head) * d_model**-0.5) for _ in range(n_heads)])
        # Similarly W_K, W_V, W_O
    
    def forward(self, x):
        out = torch.zeros_like(x)
        for h in range(self.n_heads):
            Q, K, V = x @ self.W_Q[h], x @ self.W_K[h], x @ self.W_V[h]
            attn = F.softmax((Q @ K.T) / sqrt(d_head), dim=-1)
            out += (attn @ V) @ self.W_O[h]
        return out
```

### 2.2 Input: 257 Tokens

- 256 spatial tokens (state ∈ {0,1})
- 1 CLS token (state = 2)
- Learned state embedding + learned position embedding

### 2.3 Output Heads

- **Value:** `Linear(128, 1)` on CLS → sigmoid
- **Policy:** For each move, concatenate embeddings of the two squares involved, pass through `MLP(2*d_model, d_model, 1)` → softmax over all moves
- **Sector:** `Linear(128, 16)` on CLS

### 2.4 Policy Masking

Get legal moves from `game[4]` (boolean array of length 480).
Mask illegal moves with -1e9 before softmax.

### 2.5 Tests
- [ ] Forward pass works
- [ ] Output shapes correct
- [ ] Masking correct
- [ ] Param counts match architecture (print and verify reasonable)

---

## Section 3: Data Generation Phase 1 (Alpha-Beta)

### 3.1 Simple Eval
Evaluates position from **vertical's perspective** (consistent with neural network value head):
```python
def simple_eval(game):
    v_moves = game[4][:P_MOVES].sum()  # Vertical's remaining moves
    h_moves = game[4][P_MOVES:].sum()  # Horizontal's remaining moves
    return sigmoid(0.3 * (v_moves - h_moves))  # Higher = better for vertical
```

### 3.2 Alpha-Beta
Returns value from **vertical's perspective**:
- Vertical (game[1]==False) is MAX player
- Horizontal (game[1]==True) is MIN player

**Progressive depth scheme** (d1/d2/d3):
- First 20 moves: random (no search, skip expensive opening)
- ≥90 legal moves: depth 1 (wide open positions)
- 35-89 legal moves: depth 2 (mid-game)
- <35 legal moves: depth 3 (endgame, precise calculation)

### 3.3 Data Collection & Storage

**Game Generation:**
- 30% random exploration probability during play
- Skip alpha-beta for random moves (saves computation)
- First 16 moves are random (skip expensive opening search)

**Storage Format:** Compact game records (move sequences only)
```python
# NPZ file structure:
{
    'moves': np.array((n_games, max_len), dtype=int16),  # -1 for padding
    'lengths': np.array((n_games,), dtype=int16),         # actual game lengths
    'winners': np.array((n_games,), dtype=bool)           # True = vertical won
}
```

**CLI Usage:**
```bash
python data_gen.py --n_games 25000 --output data/phase1_games.npz --seed 42
python data_gen.py --test  # Run tests only
```

**Why move sequences instead of tensors:**
- 4x smaller storage (no augmentation duplication)
- More flexible: can reconstruct any position by replay
- Augmentation applied on-the-fly during training (see Section 3.5)

### 3.4 Tests
- [x] Games terminate
- [x] Values assigned correctly (winner's moves get 1)
- [x] ~25K games in ~1 hour (8 cores) at ~1 game/sec single-threaded

### 3.5 Tensor Loader (Training-Time Data Pipeline)

The tensor loader reconstructs positions from move sequences and applies augmentation on-the-fly.

**Move Symmetry Transforms** (in `domineering_game.py`):
```python
# Precomputed lookup tables
MOVE_HFLIP[move]   # horizontal flip
MOVE_VFLIP[move]   # vertical flip
MOVE_ROT180[move]  # 180-degree rotation

# Interface functions
augment_move(move, symmetry)           # single move
augment_move_sequence(moves, symmetry) # array of moves
# symmetry: 0=identity, 1=hflip, 2=vflip, 3=rot180
```

**Position Reconstruction:**
```python
def replay_game(moves, stop_at_move, symmetry=0):
    """Replay a game to reconstruct position at a specific move."""
    # Apply symmetry to move sequence
    aug_moves = augment_move_sequence(moves[:stop_at_move], symmetry)

    # Replay moves to build game state
    game = domineering_game()
    for move in aug_moves:
        make_move(game, move)

    return game
```

**Training Batch Construction:**
```python
def make_training_batch(games_npz, batch_size):
    """Sample positions and construct training tensors."""
    # 1. Sample games and positions within games
    game_indices = sample_games(batch_size)

    for game_idx in game_indices:
        moves = games_npz['moves'][game_idx]
        length = games_npz['lengths'][game_idx]
        winner = games_npz['winners'][game_idx]

        # 2. Sample random position in game (after opening)
        pos_idx = random.randint(N_RANDOM_OPENING, length - 1)

        # 3. Sample random symmetry for augmentation
        symmetry = random.randint(0, 3)

        # 4. Replay to position and extract tensors
        game = replay_game(moves, pos_idx, symmetry)

        tokens = game_to_tokens(game)           # (257,) - board + CLS
        value = 1.0 if winner else 0.0          # from vertical's perspective
        policy = augment_move(moves[pos_idx], symmetry)  # next move played
        mask = legal_moves(game)                # (480,) boolean
        sectors = compute_sector_targets(game)  # (16,) auxiliary task
```

**Key Design Decisions:**
1. **Random symmetry per sample**: Each position gets a random augmentation, providing 4x effective data diversity
2. **Skip opening positions**: First 16 moves are random and provide little learning signal
3. **Policy target**: The move actually played (may be random or from search)
4. **Value target**: Game outcome (1.0 for vertical win, 0.0 for horizontal)

### 3.6 Tests (Tensor Loader)
- [ ] Augmented games are valid (all moves legal after transformation)
- [ ] Symmetry transforms are self-inverse (augment twice = original)
- [ ] Batch construction produces correct tensor shapes
- [ ] No game appears in both train and val/test splits

---

## Section 4: Data Generation Phase 2 (Network Bootstrap)

Train value-only network on Phase 1 data, then use for 1-ply lookahead self-play.

### Tests
- [ ] Value network trains (loss decreases)
- [ ] Network-guided games beat alpha-beta

---

## Section 5: MCTS with RAVE

### 5.1 DAG Structure
Store nodes by Zobrist hash to share transpositions.

### 5.2 UCT-RAVE Selection
```python
if n == 0:
    score = q_amaf + c * prior * sqrt(N)
else:
    beta = sqrt(k / (3*N + k))  # k=1000
    q = (1-beta)*q_mc + beta*q_amaf
    score = q + c * prior * sqrt(log(N)) / (1+n)
```

### 5.3 AMAF Update
On backprop, credit all moves played in simulation to all ancestor nodes where that move was legal.

### 5.4 Tests
- [ ] Finds obvious wins
- [ ] DAG has fewer nodes than simulations (transpositions working)
- [ ] MCTS beats alpha-beta

---

## Section 6: Training Loop

Standard PyTorch training with:
- AdamW, lr=3e-4, weight_decay=0.01
- Cosine LR schedule
- FP16 via autocast + GradScaler
- Batch size 256

Loss computation:
```python
if use_auxiliary_task:
    loss = value_loss + policy_loss + 0.25 * sector_loss
else:
    loss = value_loss + policy_loss  # No sector loss for ablation
```

### Tests
- [ ] Loss decreases
- [ ] Fits in 12GB VRAM
- [ ] Auxiliary task can be disabled

---

## Section 7: Network Embedding (CRITICAL)

### 7.1 The Embedding Operation

For layers 0-1 of large network (matching small network's 2 layers):
- **Heads 0-3:** Copy W_Q, W_K, W_V, W_O from small network
- **Heads 4-7:** Keep random W_Q, W_K, W_V; **set W_O = 0**
- **MLP neurons 0-511:** Copy W_in rows and W_out columns from small
- **MLP neurons 512-767:** Keep random W_in; **set W_out columns = 0**
- **LayerNorm:** Copy γ, β from small network

Layers 2-6 of large network: Keep fully random initialization.

Also copy: embedding layer, output heads, final LayerNorm.

### 7.2 Why W_O = 0 and W_out = 0

This makes fresh components in layers 0-1 contribute zero at initialization. Therefore:
1. Residual stream in layers 0-1 matches small network exactly
2. LayerNorm sees expected statistics
3. Pretrained components receive expected inputs
4. **Large network's embedded portion (layers 0-1) produces identical representations to small network at init**

Note: Layers 2-6 are fully random and will produce different final outputs. Verification only checks the embedded layers.

### 7.3 Verification (MUST PASS)

Verify by running only the embedded layers (0-1) of the large network:

```python
def verify_embedding(small, large, batch):
    # Small network - full forward pass
    s_v, s_p, s_s = small(batch)

    # Large network - CUT at embedded depth (only run layers 0-1)
    x = large.state_embed(batch) + large.pos_embed(large.positions)
    for layer in large.layers[:2]:  # Only first 2 layers
        x = layer(x)
    x = large.final_ln(x)

    spatial = x[:, :256]
    cls = x[:, -1]
    l_v = torch.sigmoid(large.value_head(cls))
    l_p = large.policy_head(spatial)
    l_s = large.sector_head(cls)

    assert (s_v - l_v).abs().max() < 1e-5
    assert (s_p - l_p).abs().max() < 1e-5
    assert (s_s - l_s).abs().max() < 1e-5
```

**If this fails, the experiment is invalid.** Debug until it passes.

### 7.4 Tests
- [ ] Embedding runs without error
- [ ] **verify_embedding passes**
- [ ] Fresh W_O are exactly zero
- [ ] Fresh W_out columns are exactly zero

---

## Section 8: Experiments

### 8.1 Protocol

For each seed in [42, 123, 456, 789]:

**Phase 1: Train small networks**
1. Train small network WITH auxiliary task (sector loss weight = 0.25)
2. Train small network WITHOUT auxiliary task (sector loss weight = 0.0)

**Phase 2: Train large networks**
3. Train large baseline (no embedding, with auxiliary task)
4. Create large network, embed small+aux into it, verify, train with auxiliary task
5. Create large network, embed small-no-aux into it, verify, train with auxiliary task

Total: 20 models (4 seeds × 5 conditions)

### 8.2 Key Comparisons
- **Embedding benefit:** Compare conditions 4 & 5 vs 3
- **Auxiliary task benefit:** Compare condition 4 vs 5
- **Pure parameter benefit:** Compare condition 5 vs 3

### 8.3 Tests
- [ ] All 20 runs complete
- [ ] Small networks trained with/without auxiliary task
- [ ] Embedding verification passes for conditions 4 & 5
- [ ] Results reproducible with same seed

---

## Section 8.5: Data Splitting Strategy

### Split by Game Index

To prevent data leakage, all positions from a single game must stay together in the same split. Since games are stored as move sequences in a single NPZ file, splitting is straightforward:

```python
def split_game_indices(n_games, seed=42):
    """Split game indices into train/val/test sets."""
    np.random.seed(seed)
    indices = np.arange(n_games)
    np.random.shuffle(indices)

    train_end = int(0.8 * n_games)
    val_end = int(0.9 * n_games)

    return {
        'train': indices[:train_end],
        'val': indices[train_end:val_end],
        'test': indices[val_end:]
    }
```

### Dataset Class

```python
class DomineeringDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, game_indices, positions_per_game=10):
        data = np.load(npz_path)
        self.moves = data['moves'][game_indices]
        self.lengths = data['lengths'][game_indices]
        self.winners = data['winners'][game_indices]
        self.positions_per_game = positions_per_game

    def __len__(self):
        return len(self.moves) * self.positions_per_game

    def __getitem__(self, idx):
        game_idx = idx // self.positions_per_game
        # Sample random position and symmetry
        symmetry = random.randint(0, 3)
        pos_idx = random.randint(N_RANDOM_OPENING, self.lengths[game_idx] - 1)

        # Replay and extract tensors
        game = replay_game(self.moves[game_idx], pos_idx, symmetry)
        return {
            'tokens': game_to_tokens(game),
            'value': 1.0 if self.winners[game_idx] else 0.0,
            'policy': augment_move(self.moves[game_idx, pos_idx], symmetry),
            'mask': legal_moves(game),
            'sectors': compute_sector_targets(game)
        }
```

### Tests
- [ ] No game appears in multiple splits
- [ ] 80/10/10 split ratio maintained
- [ ] Same splits used across all experimental conditions
- [ ] DataLoader produces correct batch shapes

---

## Section 9: Interpretability

### 9.1 Periodic Probe Training

Train probes throughout training, not just at the end, to track how representations evolve.

```python
def train_probe(activations, targets, alpha=1.0):
    """Train Ridge regression probe."""
    from sklearn.linear_model import Ridge
    X_train, X_val, y_train, y_val = train_test_split(
        activations, targets, test_size=0.2, random_state=42
    )
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)
    r2 = probe.score(X_val, y_val)
    return probe, r2

# During training, every 1000 steps:
def evaluate_probes(model, val_data):
    results = {}
    # For small network: probe layers 0-1
    # For large network: probe layers 0-6
    n_layers = 2 if model.n_layers == 2 else 7
    for layer_idx in range(n_layers):
        # Extract CLS activations at this layer
        activations = get_layer_activations(model, val_data, layer_idx)
        sector_targets = [d['sectors'] for d in val_data]
        
        _, r2 = train_probe(activations, sector_targets)
        results[f'probe_r2_layer_{layer_idx}'] = r2
    
    return results
```

### 9.2 What to Track

For all 3 large model variants (baseline, embedded+aux, embedded-no-aux):

1. **Probe R² by layer** every 1000 training steps
2. **Representation similarity** between embedded and baseline models
3. **Feature persistence**: Does the small network's sector representation remain in layers 0-1?

### 9.3 Expected Results

**Hypothesis 1:** In the embedded+aux model, sector information should be more concentrated at layer 1 (which was responsible for predicting the sector info in the small embedded model) throughout training.

**Hypothesis 2:** In the embedded-no-aux model, sector representations may fade over training since they're not reinforced by the loss.

**Hypothesis 3:** The embedded+aux model should maintain higher probe accuracy in early layers throughout training.

### 9.4 Implementation

```python
# In training loop
if step % 1000 == 0:
    probe_results = evaluate_probes(model, val_data)
    wandb.log({**probe_results, 'step': step})
```

### 9.5 Tests
- [ ] Before/early in training, probe at depth 2 performs best in embedded+aux, worst in non-embedded, middle in embedded+no aux. (If this doesn't happen we're probably not embedding correctly)
- [ ] Probes trained on same val split as main model
- [ ] R² logged for all layers, all model variants
- [ ] Probe results saved with checkpoints
- [ ] Visualization of R² over training time

---

## Appendix: Time Budget

| Task | Hours |
|------|-------|
| Phase 1 data (CPU) | 6-10 |
| Phase 2 training + data | 3-5 |
| Phase 3 MCTS data | 4-8 |
| Final training (20 models) | 18-24 |
| Analysis | 3 |
| **Total** | **~50** |

Fits in 5-7 days with buffer.

---

## Appendix: File Structure

```
domineering/
├── domineering_game.py # Game logic, move augmentation (MOVE_HFLIP/VFLIP/ROT180)
├── model.py            # DomineeringTransformer, game_to_tokens
├── data_gen.py         # Alpha-beta search, game generation CLI
├── data_loader.py      # DomineeringDataset, replay_game, batch construction
├── mcts.py             # DAG-MCTS with RAVE
├── training.py         # Train loop, loss functions
├── embedding.py        # embed_small_into_large, verify_embedding
├── probing.py          # Linear probes
├── experiments.py      # Main runner
├── config.py           # All hyperparameters
└── data/
    └── phase1_games.npz  # Generated game records
```

---

## Appendix A: Quick Reference

### Key Dimensions

| Name | Small | Large |
|------|-------|-------|
| d_model | 128 | 128 |
| d_head | 16 | 16 |
| n_heads | 4 | 8 |
| n_layers | 2 | 7 |
| d_mlp | 512 | 768 |
| Params | ~350K | ~2.5M |

### Weight Tensor Shapes (per head, nn.Linear convention: (out, in))

| Weight | Small Shape | Large Shape |
|--------|-------------|-------------|
| W_Q[i].weight | (16, 128) | (16, 128) |
| W_K[i].weight | (16, 128) | (16, 128) |
| W_V[i].weight | (16, 128) | (16, 128) |
| W_O[i].weight | (128, 16) | (128, 16) |
| W_in.weight | (512, 128) | (768, 128) |
| W_out.weight | (128, 512) | (128, 768) |

### Move Indexing

```python
# Moves 0-239: Vertical moves
# For vertical move m: places domino at (i,j) and (i+1,j)
i, j = divmod(m, BOARD_SIZE)

# Moves 240-479: Horizontal moves  
# For horizontal move m: places domino at (i,j) and (i,j+1)
j, i = divmod(m - P_MOVES, BOARD_SIZE)
```

### Sector Computation

```python
# 16x16 board, 4x4 sectors
sector_idx = (row // 4) * 4 + (col // 4)

# Sector (0,0) covers rows 0-3, cols 0-3
# Sector (0,1) covers rows 0-3, cols 4-7
# etc.
```

---

## Appendix B: Debugging Guide

### Loss is NaN
1. Check for division by zero in masked softmax
2. Verify policy mask is correct (has at least one True per sample)
3. Reduce learning rate
4. Check for extreme values in sector targets

### Embedding verification fails
1. Check weight tensor shapes match expected
2. Verify copy direction (src → dst, not dst → src)
3. Check that epsilon=0 (not just small)
4. Ensure fresh layers also have W_O zeroed

### MCTS is slow
1. Batch neural network evaluations
2. Reduce n_simulations for data generation
3. Check transposition table hit rate
4. Profile to find bottleneck

### Model doesn't improve
1. Verify data quality (spot check some positions)
2. Check that legal move masking is correct
3. Try training on value only first
4. Increase model size or data quantity

### Probe accuracy is low everywhere
1. Verify sector targets are computed correctly
2. Check that CLS token is being used (not spatial tokens)
3. Try probing with more features (all tokens, not just CLS)
4. Increase probe regularization