# Double Deep Q-Network (Double DQN)

## Overview

Double DQN addresses the overestimation bias inherent in standard DQN by decoupling action selection from action evaluation. This modification significantly improves learning stability and performance by reducing the positive bias in Q-value estimates.

## Algorithm Description

Double DQN uses the online network to select actions but evaluates them using the target network. This prevents the overoptimistic value estimates that occur when the same network both selects and evaluates actions.

**Key Innovation:**
- **Decoupled Action Selection/Evaluation**: Online network selects actions, target network evaluates them
- **Reduced Overestimation**: Mitigates the maximization bias of standard Q-learning

## Key Papers

- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2015) - Double DQN

## Implementation Details

This Equinox implementation ([`ff_ddqn.py`](../../../zenoqx/systems/q_learning/ff_ddqn.py)) features:
- Same network architecture as DQN (2-layer MLP with 256 hidden units)
- Modified Q-learning update using double Q-learning loss
- Experience replay buffer and target networks from DQN
- Identical hyperparameters to DQN for fair comparison

### Double Q-Learning Loss Function

The key difference from DQN is in the loss computation:

```python
def double_q_learning(
    q_tm1: chex.Array,        # Online Q-values at t-1
    q_t_value: chex.Array,    # Target Q-values at t
    a_tm1: chex.Array,        # Actions at t-1
    r_t: chex.Array,          # Rewards at t
    d_t: chex.Array,          # Discount factors
    q_t_selector: chex.Array, # Online Q-values at t (for action selection)
    huber_loss_parameter: chex.Array,
) -> jnp.ndarray:
    """Computes the double Q-learning loss."""
    batch_indices = jnp.arange(a_tm1.shape[0])
    
    # Key difference: use online network to select actions
    # but target network to evaluate them
    target_tm1 = r_t + d_t * q_t_value[batch_indices, q_t_selector.argmax(-1)]
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]
    
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)
    return jnp.mean(batch_loss)
```

### Algorithm Flow

1. **Action Selection**: Use online network to select best action: `a* = argmax_a Q_online(s', a)`
2. **Action Evaluation**: Use target network to evaluate selected action: `Q_target(s', a*)`
3. **Target Computation**: `target = r + γ * Q_target(s', a*)`
4. **Loss**: Compute TD error between online Q-value and target

## Related Algorithms

- [DQN](dqn.md) - Base algorithm
- [C51](c51.md) - Can be combined with Double DQN
- [Rainbow](rainbow.md) - Includes Double DQN as a component
