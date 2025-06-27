# Proximal Policy Optimization (PPO)

## Overview

PPO is a policy gradient method that strikes a balance between sample efficiency and implementation simplicity. It uses a clipped surrogate objective to prevent large policy updates that could destabilize training, making it more stable than basic policy gradient methods.

## Algorithm Description

PPO optimizes a clipped surrogate objective function:

- **Clipped Objective**: Limits policy updates to prevent destructive changes
- **Importance Sampling**: Uses ratio between new and old policies
- **Multiple Epochs**: Reuses collected data for multiple gradient steps
- **Advantage Estimation**: Uses GAE (Generalized Advantage Estimation)

**Key Components:**

- **Clipped Surrogate Loss**: `L^CLIP(θ) = min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)`
- **Value Function Loss**: Critic loss for advantage estimation
- **Entropy Bonus**: Encourages exploration
- **GAE**: Reduces variance in advantage estimates

## Key Papers

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) - PPO paper
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (Schulman et al., 2016) - GAE paper
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) (Schulman et al., 2015) - TRPO (PPO's predecessor)

## Implementation Details

This Equinox implementation ([`ff_ppo.py`](../../../zenoqx/systems/ppo/anakin/ff_ppo.py)) features:

Coming soon - detailed implementation specifics for the Equinox version.

## Related Algorithms

- [REINFORCE](../policy_gradient/reinforce.md) - Basic policy gradient method
- [A2C](a2c.md) - Simpler actor-critic method
- [SAC](sac.md) - Off-policy alternative
- [TRPO](trpo.md) - PPO's more complex predecessor
