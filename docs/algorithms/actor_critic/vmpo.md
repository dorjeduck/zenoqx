# V-MPO (V-trace Maximum a Posteriori Policy Optimization)

## Overview

V-MPO extends MPO to continuous control by incorporating V-trace for off-policy correction and using a continuous policy parameterization. It maintains MPO's principled EM-based optimization while handling continuous action spaces effectively.

## Algorithm Description

V-MPO adapts MPO for continuous control:

- **V-trace Correction**: Handles off-policy data with importance sampling correction
- **Continuous Policy**: Uses normal distributions for continuous actions
- **Sample-based Learning**: Works with sampled continuous actions
- **Multi-step Returns**: Uses V-trace for multi-step off-policy evaluation
- **EM Policy Optimization**: Retains MPO's principled policy improvement

**Key Components:**

- **V-trace**: Off-policy correction for value function learning
- **Gaussian Policy**: Continuous action distributions with learned variance
- **Sample-based M-step**: Fits continuous policy to weighted samples
- **Multi-step Value Learning**: V-trace enables stable off-policy learning

## Key Papers

- [V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control](https://arxiv.org/abs/1909.12238) (Song et al., 2020) - V-MPO paper
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) (Espeholt et al., 2018) - V-trace paper

## Implementation Details

This Equinox implementation ([`ff_vmpo.py`](../../../zenoqx/systems/mpo/ff_vmpo.py)) features:

## Related Algorithms

- [MPO](mpo.md) - Base algorithm for discrete actions
- [IMPALA](impala.md) - V-trace-based distributed RL
- [PPO](ppo.md) - Alternative policy optimization method
- [SAC](sac.md) - Another continuous control algorithm
