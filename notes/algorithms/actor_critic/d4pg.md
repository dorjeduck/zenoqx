# Distributed Distributional Deep Deterministic Policy Gradient (D4PG)

## Overview

D4PG extends DDPG by incorporating distributional reinforcement learning principles. Instead of learning expected Q-values, D4PG learns the full distribution of returns, providing richer information about the value function and potentially more stable learning.

## Algorithm Description

D4PG combines several improvements over DDPG:

- **Distributional Critic**: Models full return distribution instead of expected values
- **N-step Returns**: Uses multi-step returns for better credit assignment
- **Prioritized Experience Replay**: Samples important transitions more frequently
- **Multiple Distributed Actors**: Distributed data collection (in original paper)

**Key Components:**

- **Categorical Distribution**: Critic outputs probabilities over return atoms
- **Distributional Bellman Operator**: Projects target distributions onto support
- **Multi-step Returns**: Reduces variance in value estimates
- **Prioritized Replay**: Focuses learning on important experiences

## Key Papers

- [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617) (Barth-Maron et al., 2018) - D4PG paper
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) (Bellemare et al., 2017) - Distributional RL foundation

## Implementation Details

This Equinox implementation ([`ff_d4pg.py`](../../../zenoqx/systems/ddpg/ff_d4pg.py)) features:

## Related Algorithms

- [DDPG](ddpg.md) - Foundation algorithm that D4PG extends
- [C51](../q_learning/c51.md) - Distributional RL in discrete action spaces
- [QR-DQN](../q_learning/qr_dqn.md) - Alternative distributional approach
- [SAC](sac.md) - Another advanced continuous control method
