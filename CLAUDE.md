# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project investigating neural network embedding techniques using the board game Domineering as a testbed. The experiment tests whether embedding a smaller pretrained transformer into a larger one accelerates learning.

## Key Resources

- `IMPLEMENTATION_CHECKLIST.md`: Contains the complete technical specification, architecture details, and implementation roadmap. Always reference this document for specific implementation requirements.

## Coding Guidelines

### Code Style
- Use functional programming style where appropriate (see `domineering_game.py` for examples)
- Prefer numpy arrays for game state manipulation and efficiency
- Use descriptive function names that clearly indicate purpose
- Keep functions focused on single responsibilities

### Performance Considerations
- Pre-compute expensive operations where possible (e.g., move elimination tables)
- Use boolean masking for efficient array operations
- Leverage numpy vectorization instead of explicit loops

### Python Conventions
- No formal linting or formatting configuration exists yet
- Follow PEP 8 conventions for Python code
- Use type hints where they improve clarity
- Document complex algorithms with clear comments

### Development Workflow
- The project follows a checkpoint-based development approach
- Each section in the implementation checklist has specific verification tests
- Always verify correctness before moving to the next implementation phase
- No formal testing framework is in place - tests are integrated into the implementation flow

## Important Implementation Notes

1. **State Representation**: The game uses boolean arrays (True=occupied, False=empty). The neural network input is binary - which player placed a piece is strategically irrelevant since both players see the same board position.

2. **Neural Network Design**: Pay special attention to the per-head weight storage pattern - this is critical for the embedding experiment to work correctly.

3. **Verification First**: The embedding verification step (Section 7) is critical. Never proceed with training if the verification fails.

4. **No Package Management**: This project currently has no requirements.txt or similar. Dependencies (numpy, torch) need to be installed manually.

## WSL Environment

This project runs in WSL. Python with numpy/torch is available via miniforge:

```bash
# Run Python scripts
python3 script.py

# Run tests
python3 domineering_game.py
```

Required dependencies:
- numpy
- torch
- tensorboard