# AlphaZero

## Overview

AlphaZero is a general game-playing algorithm that combines Monte Carlo Tree Search (MCTS) with deep neural networks. It learns to play games from scratch through self-play, without any domain knowledge beyond the game rules, and has achieved superhuman performance in chess, shogi, and Go.

## Algorithm Description

AlphaZero uses self-play with MCTS and neural network guidance:

- **Self-Play**: Generates training data by playing against itself
- **MCTS**: Uses tree search to explore possible futures
- **Neural Network**: Predicts move probabilities and position values
- **Policy Improvement**: MCTS search improves upon neural network policy
- **Iterative Training**: Neural network learns from MCTS-improved policies

**Key Components:**

- **Neural Network**: Combined policy and value network
- **MCTS**: Tree search with neural network guidance
- **Self-Play**: Data generation through self-competition
- **No Domain Knowledge**: Learns only from game rules and self-play

## Key Papers

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (Silver et al., 2017) - AlphaZero paper
- [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) (Silver et al., 2017) - AlphaGo Zero
- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://science.sciencemag.org/content/362/6419/1140) (Silver et al., 2018) - Science paper

## Implementation Details

This Equinox implementation ([`ff_az.py`](../../../zenoqx/systems/search/ff_az.py)) features:

## Related Algorithms

- [Sampled AlphaZero](sampled_alphazero.md) - Efficiency improvements to AlphaZero
- [MuZero](muzero.md) - Model-based extension of AlphaZero
- [MCTS](mcts.md) - Core search algorithm used in AlphaZero
