# Maximum a Posteriori Policy Optimization (MPO)

## Overview

MPO is a policy optimization algorithm that uses expectation-maximization (EM) to improve policies. It decouples policy improvement from policy evaluation by using a principled approach based on KL-divergence constraints and relative rewards.

## Algorithm Description

MPO uses EM-based policy optimization:

- **E-step**: Weight samples by exponential of relative advantage
- **M-step**: Fit new policy to weighted sample distribution  
- **KL Constraints**: Limits policy updates to prevent instability
- **Temperature Parameter**: Controls the strength of policy updates
- **Dual Formulation**: Converts constrained optimization to unconstrained

**Key Components:**

- **EM Framework**: Principled policy improvement via expectation-maximization
- **Weighted Behavioral Cloning**: M-step fits policy to reweighted data
- **KL Regularization**: Prevents large policy changes
- **Lagrange Multipliers**: Automatically tune constraint weights

## Key Papers

- [Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920) (Abdolmaleki et al., 2018) - MPO paper
- [Relative Entropy Regularized Policy Iteration](https://arxiv.org/abs/1812.02256) (Nachum et al., 2017) - Related work on relative entropy methods

## Implementation Details

This Equinox implementation ([`ff_mpo.py`](../../../zenoqx/systems/mpo/ff_mpo.py)) features:

## Related Algorithms

- [V-MPO](vmpo.md) - Extension to continuous control
- [SAC](sac.md) - Another entropy-regularized method
- [PPO](ppo.md) - Alternative policy optimization approach
- [TRPO](trpo.md) - Trust region method with similar motivation
