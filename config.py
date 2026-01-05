# Configuration for Domineering Neural Network Experiment

# Architecture configurations
SMALL_CONFIG = {
    'd_model': 128,
    'd_head': 16,
    'n_heads': 4,
    'n_layers': 2,
    'd_mlp': 512
}

LARGE_CONFIG = {
    'd_model': 128,
    'd_head': 16,
    'n_heads': 8,
    'n_layers': 7,
    'd_mlp': 768
}

# Medium config for bootstrap selfplay (doesn't need to match small/large dimensions)
MEDIUM_CONFIG = {
    'd_model': 80,
    'd_head': 16,
    'n_heads': 4,
    'n_layers': 5,
    'd_mlp': 320
}

# Input/Output dimensions
N_TOKENS = 257      # 256 spatial + 1 CLS
N_STATES = 3        # 0=empty, 1=occupied, 2=CLS
N_POLICY = 480      # move outputs
N_SECTORS = 16      # auxiliary task outputs

# Training hyperparameters (for later sections)
LR = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 256

# Loss weights
# Value loss (BCE) is ~0.5-0.7, policy loss (CE over 480 classes) is ~3-4
# Weight policy down to balance contributions
POLICY_LOSS_WEIGHT = 0.2
SECTOR_LOSS_WEIGHT = 0.25
