# Deep Deterministic Policy Gradient (DDPG)

## Overview

DDPG is an off-policy actor-critic algorithm for continuous control that combines the actor-critic approach with insights from DQN. It uses a deterministic policy (actor) and learns a Q-function (critic) to guide policy improvement.

## Algorithm Description

DDPG extends DQN to continuous action spaces:

- **Deterministic Policy**: Actor outputs deterministic actions
- **Q-function Critic**: Learns action-value function like DQN
- **Experience Replay**: Reuses past experiences for sample efficiency
- **Target Networks**: Slowly-updated target networks for stability
- **Exploration Noise**: Adds noise to actions for exploration

**Key Components:**

- **Actor Network**: Deterministic policy μ(s|θ^μ)
- **Critic Network**: Q-function Q(s,a|θ^Q)
- **Target Networks**: Soft updates for both actor and critic
- **Ornstein-Uhlenbeck Noise**: Temporally correlated noise for exploration

## Key Papers

- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) (Lillicrap et al., 2016) - DDPG paper
- [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.html) (Silver et al., 2014) - Deterministic policy gradient theorem

## Implementation Details

This Equinox implementation ([`ff_ddpg.py`](../../../zenoqx/systems/ddpg/ff_ddpg.py)) features:

## Related Algorithms

- [DQN](../q_learning/dqn.md) - Value-based inspiration for DDPG
- [TD3](td3.md) - Improved version of DDPG
- [SAC](sac.md) - Stochastic alternative with entropy regularization
- [D4PG](d4pg.md) - Distributional extension of DDPG
