import numpy as np
import copy

#Functional implementation of the combinatorial game Domineering
#First player is vertical, second player is horizontal
#Players alternate placing nonoverlapping dominoes on the board
#A player loses if they cannot make a move

#setting constants

BOARD_SIZE = 16 #size of board, can be changed to any integer > 1
N_MOVES = int(2*BOARD_SIZE*(BOARD_SIZE-1)) #number of valid moves on that board
P_MOVES = int(N_MOVES/2) #number of valid moves for one player
N_SQUARES = BOARD_SIZE * BOARD_SIZE  # 256
SECTOR_SIZE = 4
N_SECTORS = (BOARD_SIZE // SECTOR_SIZE) ** 2  # 16 for 16x16 board with 4x4 sectors

ones_array = np.ones(P_MOVES,dtype=bool)
zeros_array = np.zeros(P_MOVES,dtype=bool)

player_1_legal_moves = np.concatenate([ones_array,zeros_array])
player_2_legal_moves = np.concatenate([zeros_array,ones_array])

#pre-calculate moves eliminated by each possible move

ones_array = np.ones(P_MOVES,dtype=bool)
zeros_array = np.zeros(P_MOVES,dtype=bool)

player_1_legal_moves = np.concatenate([ones_array,zeros_array])
player_2_legal_moves = np.concatenate([zeros_array,ones_array])

move_eliminations = []

for move in range(0,N_MOVES):
    valid_moves = np.ones(N_MOVES, dtype=bool)

    if move<P_MOVES:
        i,j = divmod(move,BOARD_SIZE)
        equivalent_move = BOARD_SIZE*j + i + P_MOVES
        p1_eliminated_moves = [move-BOARD_SIZE,move,move+BOARD_SIZE]
        p2_eliminated_moves = [equivalent_move - BOARD_SIZE, equivalent_move - BOARD_SIZE+1, equivalent_move, equivalent_move+1]
        for m in p1_eliminated_moves:
            if 0 <= m < P_MOVES:
                valid_moves[m] = False
        for m in p2_eliminated_moves:
            if P_MOVES <= m < N_MOVES:
                valid_moves[m] = False
    else:
        j,i = divmod(move-P_MOVES,BOARD_SIZE)
        equivalent_move = BOARD_SIZE*i + j
        p1_eliminated_moves = [equivalent_move - BOARD_SIZE, equivalent_move - BOARD_SIZE+1, equivalent_move, equivalent_move+1]
        p2_eliminated_moves = [move-BOARD_SIZE,move,move+BOARD_SIZE]
        for m in p1_eliminated_moves:
            if 0 <= m < P_MOVES:
                valid_moves[m] = False
        for m in p2_eliminated_moves:
            if P_MOVES <= m < N_MOVES:
                valid_moves[m] = False

    move_eliminations.append(valid_moves)

# Move symmetry transformation lookup tables
# symmetry: 0=identity, 1=hflip, 2=vflip, 3=rot180
MOVE_HFLIP = np.zeros(N_MOVES, dtype=np.int16)
MOVE_VFLIP = np.zeros(N_MOVES, dtype=np.int16)
MOVE_ROT180 = np.zeros(N_MOVES, dtype=np.int16)

for m in range(P_MOVES):  # Vertical moves
    i, j = divmod(m, BOARD_SIZE)
    MOVE_HFLIP[m] = i * BOARD_SIZE + (BOARD_SIZE - 1 - j)
    MOVE_VFLIP[m] = (BOARD_SIZE - 2 - i) * BOARD_SIZE + j
    MOVE_ROT180[m] = (BOARD_SIZE - 2 - i) * BOARD_SIZE + (BOARD_SIZE - 1 - j)

for m in range(P_MOVES, N_MOVES):  # Horizontal moves
    j, i = divmod(m - P_MOVES, BOARD_SIZE)
    MOVE_HFLIP[m] = P_MOVES + (BOARD_SIZE - 2 - j) * BOARD_SIZE + i
    MOVE_VFLIP[m] = P_MOVES + j * BOARD_SIZE + (BOARD_SIZE - 1 - i)
    MOVE_ROT180[m] = P_MOVES + (BOARD_SIZE - 2 - j) * BOARD_SIZE + (BOARD_SIZE - 1 - i)

# Move-to-squares lookup: which 2 square indices each move fills
move_to_squares = np.zeros((N_MOVES, 2), dtype=np.int16)
for m in range(P_MOVES):  # Vertical moves fill (i,j) and (i+1,j)
    i, j = divmod(m, BOARD_SIZE)
    move_to_squares[m] = [i * BOARD_SIZE + j, (i + 1) * BOARD_SIZE + j]
for m in range(P_MOVES, N_MOVES):  # Horizontal moves fill (i,j) and (i,j+1)
    j, i = divmod(m - P_MOVES, BOARD_SIZE)
    move_to_squares[m] = [i * BOARD_SIZE + j, i * BOARD_SIZE + (j + 1)]

# Zobrist hashing for transposition detection
np.random.seed(42)  # For reproducibility
zobrist_table = np.random.randint(0, 2**63, size=(N_SQUARES, 2), dtype=np.uint64)
zobrist_player = np.random.randint(0, 2**63, dtype=np.uint64)

def compute_hash(game):
    """Compute Zobrist hash for current game state."""
    h = np.uint64(0)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if game[0][i, j]:  # If square is occupied
                h ^= zobrist_table[i * BOARD_SIZE + j, 1]
            else:  # If square is empty
                h ^= zobrist_table[i * BOARD_SIZE + j, 0]
    if game[1]:  # If player 1 (horizontal) to move
        h ^= zobrist_player
    return h

# Precompute sector move masks for auxiliary task
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

def compute_sector_targets(game):
    """Returns (16,) array of (v_moves - h_moves) per 4×4 sector."""
    legal_moves = game[4]  # Boolean array of length 480
    targets = np.zeros(N_SECTORS, dtype=np.float32)
    
    for sector_idx in range(N_SECTORS):
        v_count = (legal_moves & sector_v_masks[sector_idx]).sum()
        h_count = (legal_moves & sector_h_masks[sector_idx]).sum()
        targets[sector_idx] = v_count - h_count
    
    return targets

#Implementing the game

def domineering_game():
    #returns a list of all the data required for a Domineering game
    state = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=bool)
    player = False #player 0, boolean
    winner = None
    done = False
    remaining_moves = np.ones((N_MOVES,),dtype=bool)
    history = []
    return [state,player,winner,done,remaining_moves,history]

def reset(game):
    state = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=bool)
    remaining_moves = np.ones(N_MOVES,dtype=bool)
    history = []
    game[0] = state
    game[1] = False
    game[2] = None
    game[3] = False
    game[4] = remaining_moves
    game[5] = history

def representation(game):
    return np.copy(game[0])

def legal_moves(game):
    # return a list of legal moves for the current player
    if not game[1]:  # If current player is False (player 0)
        return game[4] & player_1_legal_moves
    else:  # If current player is True (player 1)
        return game[4] & player_2_legal_moves

def make_move(game, move):
    # Check if the move is legal
    if not legal_moves(game)[move]:
        return False

    i = None
    j = None

    if not game[1]:
        #game[1] = False = 0, so vertical player
        i,j = divmod(move,BOARD_SIZE)
    else:
        #horizontal player
        j,i = divmod(move-P_MOVES,BOARD_SIZE)

    #place the dommino
    if not game[1]:  # Player 0 (vertical)
        game[0][i, j] = True
        game[0][i+1, j] = True
    else:  # Player 1 (horizontal)
        game[0][i, j] = True
        game[0][i, j+1] = True

    #eliminate moves
    game[4] = game[4] & move_eliminations[move]

    # Append move to history
    game[5].append(move)

    # Switch the player
    game[1] = not game[1]

    # Check if the game is over
    if not any(legal_moves(game)):
        game[2] = not game[1]  # Set the winner to the player before the last move, so True if first and False if second
        game[3] = True #game is done

    return True

def copy_game(game):
    state_copy = np.copy(game[0])
    player_copy = game[1]  # Direct copy is fine for a boolean
    winner_copy = game[2]  # Direct copy is fine for a simple data type like None or boolean
    done_copy = game[3] #Direct copy is fine for a boolean
    remaining_moves_copy = np.copy(game[4])
    history_copy = game[5][:]  # Shallow copy of the list is enough if it contains only primitives
    return [state_copy, player_copy, winner_copy, done_copy, remaining_moves_copy, history_copy]

def display(game):
    state, player, winner, _, _, _ = game
    header_row = ' '
    for i in range(BOARD_SIZE):
        header_row = header_row + " " + str(i%10)
    print(header_row)
    for i in range(BOARD_SIZE):
        print(f'{i%10} ', end='')
        for j in range(BOARD_SIZE):
            if state[i, j]:
                print('X', end=' ')
            else:
                print('_', end=' ')
        print()
    print(f"Player to move: {'0' if not player else '1'}")
    if winner is not None:
        print(f"Winner: {'0' if not winner else '1'}")
    else:
        print("No winner yet")

def augment_move(move: int, symmetry: int) -> int:
    """Transform a single move index by symmetry (0=id, 1=hflip, 2=vflip, 3=rot180)."""
    if symmetry == 0:
        return move
    elif symmetry == 1:
        return MOVE_HFLIP[move]
    elif symmetry == 2:
        return MOVE_VFLIP[move]
    else:
        return MOVE_ROT180[move]

def augment_move_sequence(moves, symmetry: int):
    """Transform move indices by symmetry. Accepts int, list, or numpy array."""
    if symmetry == 0:
        return moves
    lookup = [None, MOVE_HFLIP, MOVE_VFLIP, MOVE_ROT180][symmetry]
    if isinstance(moves, (int, np.integer)):
        return int(lookup[moves])
    return lookup[np.asarray(moves)]

# ============================================================================
# Section 1.5 Tests
# ============================================================================

def test_game_plays_to_completion():
    """Test that 16x16 games terminate properly with random play."""
    import random
    random.seed(42)

    for game_num in range(5):
        game = domineering_game()
        move_count = 0
        max_moves = 500  # Safety limit

        while not game[3] and move_count < max_moves:
            legal = legal_moves(game)
            legal_indices = np.where(legal)[0]
            if len(legal_indices) == 0:
                break
            move = random.choice(legal_indices)
            make_move(game, move)
            move_count += 1

        assert game[3], f"Game {game_num} did not terminate after {move_count} moves"
        assert game[2] is not None, f"Game {game_num} has no winner"
        assert move_count < max_moves, f"Game {game_num} exceeded max moves"

    print("PASS: test_game_plays_to_completion")

def test_zobrist_transposition_consistency():
    """Test that same position reached via different move orders has same hash."""
    # Create two games and reach the same position via different move orders
    game1 = domineering_game()
    game2 = domineering_game()

    # Game 1: Play vertical move at (0,0), then vertical move at (2,0)
    # Move 0 = vertical at row 0, col 0
    # Move 32 = vertical at row 2, col 0 (2*16 = 32)
    make_move(game1, 0)   # V places at (0,0)-(1,0)
    make_move(game1, P_MOVES)  # H places at (0,0)-(0,1) - but wait, (0,0) is occupied

    # Let's use moves that don't conflict
    game1 = domineering_game()
    game2 = domineering_game()

    # Game 1: V at (0,0), H at (0,2), V at (0,4), H at (0,6)
    # Game 2: V at (0,4), H at (0,6), V at (0,0), H at (0,2)

    # V move at col j: move index = j (for row 0)
    # H move at col j: move index = P_MOVES + j (for row 0)

    v_move_0 = 0    # V at (0,0)-(1,0)
    v_move_4 = 4    # V at (0,4)-(1,4)
    h_move_2 = P_MOVES + 2  # H at (0,2)-(0,3)
    h_move_6 = P_MOVES + 6  # H at (0,6)-(0,7)

    # Game 1 order: v0, h2, v4, h6
    make_move(game1, v_move_0)
    make_move(game1, h_move_2)
    make_move(game1, v_move_4)
    make_move(game1, h_move_6)

    # Game 2 order: v4, h6, v0, h2
    make_move(game2, v_move_4)
    make_move(game2, h_move_6)
    make_move(game2, v_move_0)
    make_move(game2, h_move_2)

    # Both should have same board state
    assert np.array_equal(game1[0], game2[0]), "Board states differ"
    assert game1[1] == game2[1], "Player to move differs"

    # Zobrist hashes should match
    hash1 = compute_hash(game1)
    hash2 = compute_hash(game2)
    assert hash1 == hash2, f"Zobrist hashes differ: {hash1} vs {hash2}"

    print("PASS: test_zobrist_transposition_consistency")

def test_sector_targets_hand_crafted():
    """Test sector targets on known positions."""
    game = domineering_game()

    targets = compute_sector_targets(game)

    # All targets should be finite numbers
    assert np.all(np.isfinite(targets)), "Sector targets contain non-finite values"

    # On a fresh 16x16 board with 4x4 sectors:
    # Interior sectors (not on right/bottom edge) have 16 v-moves and 16 h-moves each
    # Sector 0 should have target = 0 (16 - 16)
    assert targets[0] == 0, f"Sector 0 should be balanced on fresh board, got {targets[0]}"

    # Sector 3 (top-right corner, cols 12-15) has fewer h-moves due to board edge
    # 16 v-moves but only 12 h-moves (can't start horizontal move at col 15)
    assert targets[3] == 4, f"Sector 3 should have v advantage of 4, got {targets[3]}"

    # Verify sector masks have correct counts on fresh board
    v_count_s0 = (game[4] & sector_v_masks[0]).sum()
    h_count_s0 = (game[4] & sector_h_masks[0]).sum()
    assert v_count_s0 == 16, f"Sector 0 should have 16 v-moves, got {v_count_s0}"
    assert h_count_s0 == 16, f"Sector 0 should have 16 h-moves, got {h_count_s0}"

    # Play a move and verify counts decrease appropriately
    make_move(game, 0)  # V at (0,0)-(1,0)

    v_count_after = (game[4] & sector_v_masks[0]).sum()
    h_count_after = (game[4] & sector_h_masks[0]).sum()

    # Move 0 eliminates v-moves at (0,0) and (1,0), and h-moves at (0,0) and (1,0)
    assert v_count_after == 14, f"After move, sector 0 should have 14 v-moves, got {v_count_after}"
    assert h_count_after == 14, f"After move, sector 0 should have 14 h-moves, got {h_count_after}"

    print("PASS: test_sector_targets_hand_crafted")

def test_move_augmentation():
    """Comprehensive tests for move symmetry transformations."""

    # 1. Self-inverse property (all transforms are involutions)
    for m in range(N_MOVES):
        assert MOVE_HFLIP[MOVE_HFLIP[m]] == m, f"hflip not self-inverse at {m}"
        assert MOVE_VFLIP[MOVE_VFLIP[m]] == m, f"vflip not self-inverse at {m}"
        assert MOVE_ROT180[MOVE_ROT180[m]] == m, f"rot180 not self-inverse at {m}"
    print("  Self-inverse: PASS")

    # 2. Composition: hflip(vflip(m)) == rot180(m)
    for m in range(N_MOVES):
        hv = MOVE_HFLIP[MOVE_VFLIP[m]]
        vh = MOVE_VFLIP[MOVE_HFLIP[m]]
        r = MOVE_ROT180[m]
        assert hv == r, f"hflip(vflip({m}))={hv} != rot180({m})={r}"
        assert vh == r, f"vflip(hflip({m}))={vh} != rot180({m})={r}"
    print("  Composition hflip(vflip)=rot180: PASS")

    # 3. Range preservation: vertical stays vertical, horizontal stays horizontal
    for m in range(P_MOVES):  # Vertical moves
        assert 0 <= MOVE_HFLIP[m] < P_MOVES, f"hflip({m}) left vertical range"
        assert 0 <= MOVE_VFLIP[m] < P_MOVES, f"vflip({m}) left vertical range"
        assert 0 <= MOVE_ROT180[m] < P_MOVES, f"rot180({m}) left vertical range"
    for m in range(P_MOVES, N_MOVES):  # Horizontal moves
        assert P_MOVES <= MOVE_HFLIP[m] < N_MOVES, f"hflip({m}) left horizontal range"
        assert P_MOVES <= MOVE_VFLIP[m] < N_MOVES, f"vflip({m}) left horizontal range"
        assert P_MOVES <= MOVE_ROT180[m] < N_MOVES, f"rot180({m}) left horizontal range"
    print("  Range preservation: PASS")

    # 4. Boundary vertical moves (row 0 and row 14)
    # Vertical move m = i*16 + j spans rows i and i+1
    # Move at (0,0): vflip should go to (14,0), not (15,0)
    m_top_left = 0 * 16 + 0  # = 0
    m_bottom_left = 14 * 16 + 0  # = 224
    assert MOVE_VFLIP[m_top_left] == m_bottom_left, \
        f"vflip of top-left vertical should be bottom-left: {MOVE_VFLIP[m_top_left]} != {m_bottom_left}"
    assert MOVE_VFLIP[m_bottom_left] == m_top_left, \
        f"vflip of bottom-left vertical should be top-left"

    # Move at (0,15): hflip should go to (0,0)
    m_top_right = 0 * 16 + 15  # = 15
    assert MOVE_HFLIP[m_top_right] == m_top_left, \
        f"hflip of (0,15) should be (0,0): {MOVE_HFLIP[m_top_right]} != {m_top_left}"
    print("  Boundary vertical moves: PASS")

    # 5. Boundary horizontal moves (col 0 and col 14)
    # Horizontal move m = 240 + j*16 + i spans cols j and j+1
    # Move at col 0, row 0: hflip should go to col 14, row 0
    m_left_top = P_MOVES + 0 * 16 + 0  # = 240
    m_right_top = P_MOVES + 14 * 16 + 0  # = 464
    assert MOVE_HFLIP[m_left_top] == m_right_top, \
        f"hflip of left-top horizontal should be right-top: {MOVE_HFLIP[m_left_top]} != {m_right_top}"

    # Move at col 0, row 15: vflip should go to col 0, row 0
    m_left_bottom = P_MOVES + 0 * 16 + 15  # = 255
    assert MOVE_VFLIP[m_left_bottom] == m_left_top, \
        f"vflip of left-bottom horizontal should be left-top: {MOVE_VFLIP[m_left_bottom]} != {m_left_top}"
    print("  Boundary horizontal moves: PASS")

    # 6. Center move should map to itself under rot180
    # Vertical move at (7, 7) spans rows 7-8, cols 7
    # rot180: (7,7) -> (14-7, 15-7) = (7, 8) -- NOT same, asymmetric board
    # Vertical move at (7, 7.5) doesn't exist, so no fixed point for rot180
    # Just verify a specific known transformation
    m_center_v = 7 * 16 + 7  # = 119
    m_center_v_rot = (14 - 7) * 16 + (15 - 7)  # = 7*16 + 8 = 120
    assert MOVE_ROT180[m_center_v] == m_center_v_rot, \
        f"rot180 of vertical (7,7) should be (7,8): {MOVE_ROT180[m_center_v]} != {m_center_v_rot}"
    print("  Center move transformation: PASS")

    # 7. Verify augment_move_sequence works
    moves = np.array([0, 119, 240, 255])
    for sym in range(4):
        result = augment_move_sequence(moves, sym)
        assert len(result) == len(moves), f"augment_move_sequence changed length for sym={sym}"
        for i, m in enumerate(moves):
            expected = augment_move(m, sym)
            assert result[i] == expected, f"augment_move_sequence mismatch at {i} for sym={sym}"
    print("  augment_move_sequence: PASS")

    print("PASS: test_move_augmentation")

def run_section_1_tests():
    """Run all Section 1.5 tests."""
    print("=" * 60)
    print("Running Section 1.5 Tests")
    print("=" * 60)

    test_game_plays_to_completion()
    test_zobrist_transposition_consistency()
    test_sector_targets_hand_crafted()
    test_move_augmentation()

    print("=" * 60)
    print("All Section 1.5 tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    run_section_1_tests()