# REINFORCE

## Overview

REINFORCE is a fundamental policy gradient algorithm that directly optimizes the policy by following the gradient of expected returns. It uses Monte Carlo sampling to estimate policy gradients and forms the foundation for many modern policy-based methods.

## Algorithm Description

REINFORCE optimizes the policy by following the policy gradient:

- **Policy Gradient**: `∇θ J(θ) = E[∇θ log π(a|s) * R(τ)]`
- **Monte Carlo Returns**: Uses complete episode returns for gradient estimation
- **Direct Policy Optimization**: Updates policy parameters directly
- **High Variance**: Suffers from high variance in gradient estimates

**Key Components:**

- **Policy Network**: Parameterized policy π(a|s;θ)
- **Return Computation**: Monte Carlo estimation of returns
- **Policy Gradient**: REINFORCE gradient estimator
- **Baseline**: Optional value function to reduce variance

## Key Papers

- [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) (Williams, 1992) - Original REINFORCE paper
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) (Sutton et al., 2000) - Policy gradient theorem

## Implementation Details

This Equinox implementation ([`ff_reinforce.py`](../../../zenoqx/systems/vpg/ff_reinforce.py)) features:

Coming soon - detailed implementation specifics for the Equinox version.

## Related Algorithms

- [PPO](../actor_critic/ppo.md) - Advanced policy gradient method
- [A2C/A3C](../actor_critic/a2c.md) - Actor-critic extension
- [SAC](../actor_critic/sac.md) - Entropy-regularized policy gradient
