# Quantile Regression DQN (QR-DQN)

## Overview

QR-DQN learns return distributions by directly estimating quantiles using quantile regression. Unlike C51's fixed categorical approach, QR-DQN provides a more flexible representation of return distributions without requiring predefined support atoms.

## Algorithm Description

QR-DQN estimates the cumulative distribution function (CDF) of returns by learning quantiles at fixed probability levels. The network outputs quantile values that minimize the quantile regression loss.

**Key Components:**

- **Quantile Regression**: Estimates quantiles of the return distribution
- **Flexible Distribution**: No fixed support atoms required
- **Quantile Huber Loss**: Asymmetric loss function for quantile estimation

## Key Papers

- [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) (Dabney et al., 2017) - QR-DQN

## Implementation Details

This Equinox implementation ([`ff_qr_dqn.py`](../../../zenoqx/systems/q_learning/ff_qr_dqn.py)) features:

- Quantile network head outputting quantile values
- Quantile regression loss with Huber smoothing
- Distributional Bellman operator for quantiles
- Risk-sensitive action selection using quantiles

### Quantile Network Architecture

```python
# TODO: Add quantile network structure
```

### Quantile Regression Loss

```python
# TODO: Add quantile regression loss computation
```

## Related Algorithms

- [DQN](dqn.md) - Base value-based method
- [C51](c51.md) - Alternative distributional approach
- [Rainbow](rainbow.md) - Could incorporate QR-DQN
