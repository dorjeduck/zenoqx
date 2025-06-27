# Munchausen DQN (MDQN)

## Overview

Munchausen DQN improves upon DQN by incorporating the scaled log-policy of the current action into the immediate reward. This modification acts as an implicit entropy regularization, encouraging exploration while maintaining the value-based approach of DQN.

## Algorithm Description

MDQN modifies the standard DQN by adding a scaled log-policy term to the reward:
- **Modified Reward**: `r'(s,a) = r(s,a) + α * log π(a|s)`
- **Entropy Regularization**: The log-policy term implicitly encourages exploration
- **Soft Greedy Policy**: Uses tau-scaled softmax policy instead of epsilon-greedy
- **Same Network Architecture**: Uses standard DQN network with modified training

**Key Components:**
- **Implicit Entropy**: Log-policy term added to reward for implicit entropy regularization
- **Soft Policy**: Tau-scaled softmax policy for action selection
- **Modified Bellman Operator**: Incorporates the scaled log-policy term

## Key Papers

- [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430) (Vieillard et al., 2020) - Original Munchausen DQN paper

## Implementation Details

This Equinox implementation ([`ff_mdqn.py`](../../../zenoqx/systems/q_learning/ff_mdqn.py)) features:

Coming soon - detailed implementation specifics for the Equinox version.

## Related Algorithms

- [DQN](dqn.md) - Base algorithm that MDQN extends
- [Double DQN](double_dqn.md) - Can be combined with Munchausen modification
- [Rainbow](rainbow.md) - Multi-component DQN improvement
- [SAC](../actor_critic/sac.md) - Another entropy-regularized method
