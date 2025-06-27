# Soft Actor-Critic (SAC)

## Overview

SAC is an off-policy actor-critic algorithm that optimizes a stochastic policy in an off-policy way. It incorporates entropy regularization to encourage exploration and uses automatic temperature tuning to balance exploration and exploitation automatically.

## Algorithm Description

SAC maximizes both expected return and policy entropy:

- **Entropy Regularization**: `J(π) = E[R(τ) + α H(π(·|s_t))]`
- **Off-policy Learning**: Uses replay buffer for sample efficiency  
- **Soft Bellman Equations**: Incorporates entropy into value functions
- **Automatic Temperature Tuning**: Adapts entropy weight α automatically
- **Double Q-learning**: Uses two Q-functions to reduce overestimation

**Key Components:**

- **Soft Q-functions**: Q-values include entropy terms
- **Stochastic Policy**: Outputs action distributions, not deterministic actions
- **Temperature Parameter**: Controls exploration/exploitation trade-off
- **Reparameterization Trick**: Enables backpropagation through stochastic policy

## Key Papers

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018) - SAC v1
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) (Haarnoja et al., 2019) - SAC v2 with automatic temperature tuning

## Implementation Details

This Equinox implementation ([`ff_sac.py`](../../../zenoqx/systems/sac/ff_sac.py)) features:

## Related Algorithms

- [DDPG](ddpg.md) - Deterministic policy gradient ancestor
- [TD3](td3.md) - Alternative off-policy method
- [PPO](ppo.md) - On-policy alternative
- [MPO](mpo.md) - Related maximum entropy method
