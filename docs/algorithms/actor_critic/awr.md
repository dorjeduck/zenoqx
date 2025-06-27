# Advantage Weighted Regression (AWR)

## Overview

AWR is a simple yet effective policy optimization algorithm that uses advantage-weighted regression to improve policies. It combines the simplicity of behavioral cloning with the effectiveness of advantage-based weighting to achieve stable policy learning.

## Algorithm Description

AWR uses advantage-weighted behavioral cloning:

- **Advantage Weighting**: Weight actions by exponential of advantage
- **Behavioral Cloning**: Fit policy to advantage-weighted action distribution
- **Value Function Learning**: Standard TD learning for advantage computation
- **Simple Implementation**: No complex objectives or constraints
- **Stable Learning**: Avoids policy collapse through implicit regularization

**Key Components:**

- **Advantage Computation**: A(s,a) = Q(s,a) - V(s)
- **Exponential Weighting**: w = exp(A(s,a) / β)
- **Weighted Regression**: Maximize weighted log-likelihood of actions
- **Temperature Parameter**: β controls the strength of advantage weighting

## Key Papers

- [Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning](https://arxiv.org/abs/1910.00177) (Peng et al., 2019) - AWR paper

## Implementation Details

This Equinox implementation ([`ff_awr.py`](../../../zenoqx/systems/awr/ff_awr.py)) features:

## Related Algorithms

- [SAC](sac.md) - Another off-policy continuous control method
- [PPO](ppo.md) - On-policy alternative
- [DDPG](ddpg.md) - Deterministic policy gradient approach
- [MPO](mpo.md) - Similar EM-based approach
