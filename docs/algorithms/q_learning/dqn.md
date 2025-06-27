# Deep Q-Network (DQN)

## Overview

Deep Q-Network (DQN) combines Q-learning with deep neural networks to enable reinforcement learning in high-dimensional state spaces. The key innovations are experience replay to break temporal correlations and a target network to stabilize training by providing consistent Q-value targets.

## Algorithm Description

DQN learns a Q-function Q(s,a) that estimates the expected cumulative reward for taking action `a` in state `s`. The agent selects actions using an ε-greedy policy based on the Q-values, and the network is trained using the temporal difference (TD) error between predicted and target Q-values.

## Key Papers

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013) - Original DQN paper
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015) - Refined DQN with target networks

## Implementation Details

This implementation ([`ff_dqn.py`](../../../zenoqx/systems/q_learning/ff_dqn.py)) features:

- Feed-forward neural network architecture
- ε-greedy action selection
- Experience replay buffer using [FlashBax](https://github.com/instadeepai/flashbax)
- Polyak averaging for target network
- Huber loss for robust Q-learning updates
- Gradient clipping

### Network Architecture

The default network uses a 2-layer MLP with 256 hidden units:

```python
# Network configuration (from mlp_dqn.yaml)
# See: zenoqx/configs/network/mlp_dqn.yaml
actor_network:
  pre_torso:
    _target_: zenoqx.networks.torso.MLPTorso
    input_dim: ~
    layer_sizes: [256, 256]
    use_layer_norm: False
    activation: silu
  action_head:
    _target_: zenoqx.networks.heads.DiscreteQNetworkHead
    input_dim: ~
    action_dim: ~
```

### Epsilon Greedy

Epsilon-greedy exploration is implemented as part of the [`DiscreteQNetworkHead`](../../../zenoqx/networks/heads.py). Note that epsilon decay is not yet implemented - epsilon remains constant during training.

```python
class DiscreteQNetworkHead:
    ...
    def __call__(self, embedding: jnp.ndarray) -> distrax.EpsilonGreedy:
        q_values = self.linear(embedding)
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)
```

### Loss Function

The Q-learning loss supports both MSE and Huber loss: ([`loss.py`](../../../zenoqx/utils/loss.py))

```python
def q_learning(
    q_tm1: chex.Array,     # Q-values at time t-1
    a_tm1: chex.Array,     # Actions at time t-1
    r_t: chex.Array,       # Rewards at time t
    d_t: chex.Array,       # Discount factors
    q_t: chex.Array,       # Q-values at time t
    huber_loss_parameter: chex.Array,
) -> jnp.ndarray:
    """Computes the Q-learning loss."""
    batch_indices = jnp.arange(a_tm1.shape[0])
    # Compute Q-learning TD target
    target_tm1 = r_t + d_t * jnp.max(q_t, axis=-1)
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]
    
    # Apply Huber or MSE loss
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)
    return jnp.mean(batch_loss)
```

### Target Network Updates

Polyak averaging for target network using Optax incremental update:

```python
new_target_q_model = optax.incremental_update(
    q_new_online_model, models.target, config.system.tau
)
```

### Gradient clipping

([`ff_dqn.py`](../../../zenoqx/systems/q_learning/ff_dqn.py))

```python
q_optim = optax.chain(
    optax.clip_by_global_norm(config.system.max_grad_norm),  
    optax.adam(q_lr, eps=1e-5),
)
```

## Related Algorithms

- [Double DQN](double_dqn.md) - Addresses overestimation bias
- [C51](c51.md) - Distributional extension of DQN
- [Rainbow](rainbow.md) - Combines multiple DQN improvements
