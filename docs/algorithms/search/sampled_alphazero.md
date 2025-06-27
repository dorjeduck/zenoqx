# Sampled AlphaZero

## Overview

Sampled AlphaZero is an efficiency improvement to AlphaZero that reduces the computational cost of MCTS by using sampling techniques. It maintains the core AlphaZero approach while making it more practical for resource-constrained environments.

## Algorithm Description

Sampled AlphaZero optimizes MCTS efficiency:

- **Sampled MCTS**: Reduces computational cost through sampling
- **Efficient Tree Search**: Maintains search quality with fewer simulations
- **Same Neural Network**: Uses standard AlphaZero network architecture
- **Reduced Computation**: Achieves similar performance with less compute
- **Practical Implementation**: More feasible for limited computational resources

**Key Components:**

- **Sampling Strategy**: Efficiently samples nodes during MCTS
- **Reduced Simulations**: Achieves good performance with fewer MCTS rollouts
- **Maintained Quality**: Preserves AlphaZero's learning effectiveness
- **Computational Efficiency**: Lower resource requirements

## Key Papers

- [Sampled AlphaZero](https://arxiv.org/abs/1902.07805) (Huizinga et al., 2019) - Sampled AlphaZero improvements
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) (Silver et al., 2017) - Original AlphaZero

## Implementation Details

This Equinox implementation ([`ff_sampled_az.py`](../../../zenoqx/systems/search/ff_sampled_az.py)) features:

## Related Algorithms

- [AlphaZero](alphazero.md) - Base algorithm that Sampled AlphaZero improves
- [MuZero](muzero.md) - Model-based extension
- [MCTS](mcts.md) - Underlying search algorithm
