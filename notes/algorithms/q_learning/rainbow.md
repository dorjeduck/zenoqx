# Rainbow DQN

## Overview

Rainbow DQN combines six key improvements to DQN into a single algorithm: Double DQN, Prioritized Experience Replay, Dueling networks, Multi-step learning, Distributional RL (C51), and Noisy networks. This combination achieves state-of-the-art performance on Atari games.

## Algorithm Description

Rainbow integrates multiple DQN enhancements:

- **Double DQN**: Reduces overestimation bias in Q-learning
- **Prioritized Replay**: Samples important transitions more frequently
- **Dueling Networks**: Separates state value and advantage estimation
- **Multi-step Returns**: Uses n-step returns for better credit assignment
- **Distributional RL**: Models full return distribution (C51)
- **Noisy Networks**: Replaces epsilon-greedy with parametric noise

**Key Components:**

- **Six DQN Improvements**: Combines the most effective DQN enhancements
- **Ablation Studies**: Shows each component contributes to performance
- **Synergistic Effects**: Components work better together than individually

## Key Papers

- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) (Hessel et al., 2018) - Rainbow DQN
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (Schaul et al., 2016) - Prioritized replay
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) (Wang et al., 2016) - Dueling networks
- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) (Fortunato et al., 2018) - Noisy networks

## Implementation Details

This Equinox implementation ([`ff_rainbow.py`](../../../zenoqx/systems/q_learning/ff_rainbow.py)) features:

Coming soon - detailed implementation specifics for the Equinox version.

## Related Algorithms

- [DQN](dqn.md) - Base algorithm
- [Double DQN](double_dqn.md) - One of Rainbow's components
- [C51](c51.md) - Distributional component of Rainbow
- [QR-DQN](qr_dqn.md) - Alternative distributional approach
