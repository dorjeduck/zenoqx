# zenoqx

Reinforcement learning with [JAX](https://jax.readthedocs.io/)/[Equinox](https://github.com/patrick-kidger/equinox). Currently focusing on porting [Stoix](https://github.com/EdanToledo/Stoix) from [Flax/Linen](https://flax-linen.readthedocs.io/en/latest/) to Equinox as a starting point.

## Purpose

Personal learning project exploring RL/JAX/Equinox. This evaluates [Equinox](https://github.com/patrick-kidger/equinox) as an alternative to the still widely used [Flax Linen](https://flax-linen.readthedocs.io/en/latest/), whose successor [NNX](https://flax.readthedocs.io/en/latest/nnx/index.html) hasn't clicked for me yet.

Beyond learning, this implementation hopefully will serve as a foundation for experimenting with novel RL approaches once it matures.

## Status

**Alpha** - Experimental, buggy, not production ready.

## Ported Algorithms (Work in Progress)

### Value-Based Methods

- **[Q-Learning](zenoqx/systems/q_learning/)**:
  - [Deep Q-Network (DQN)](zenoqx/systems/q_learning/ff_dqn.py)
  - [Double DQN](zenoqx/systems/q_learning/ff_ddqn.py)
  - [DQN with regularization](zenoqx/systems/q_learning/ff_dqn_reg.py)
  - [Munchausen DQN](zenoqx/systems/q_learning/ff_mdqn.py)
  - [Categorical DQN (C51)](zenoqx/systems/q_learning/ff_c51.py)
  - [Quantile Regression DQN](zenoqx/systems/q_learning/ff_qr_dqn.py)
  - [Rainbow DQN](zenoqx/systems/q_learning/ff_rainbow.py)

### Policy-Based Methods

- **[Policy Gradient](zenoqx/systems/vpg/)**:
  - [REINFORCE](zenoqx/systems/vpg/ff_reinforce.py)
  - [REINFORCE (continuous action space)](zenoqx/systems/vpg/ff_reinforce_continuous.py)
- **[Maximum a Posteriori Policy Optimisation (MPO)](zenoqx/systems/mpo/)**:
  - [MPO](zenoqx/systems/mpo/ff_mpo.py)
  - [V-MPO (on-policy variant)](zenoqx/systems/mpo/ff_vmpo.py)

### Actor-Critic Methods

- **[Advantage-Weighted Regression (AWR)](zenoqx/systems/awr/)**: 
  - [AWR](zenoqx/systems/awr/ff_awr.py)
- **[Deep Deterministic Policy Gradient (DDPG)](zenoqx/systems/ddpg/)**:
  - [DDPG](zenoqx/systems/ddpg/ff_ddpg.py)
  - [Distributed Distributional DDPG (D4PG)](zenoqx/systems/ddpg/ff_d4pg.py)
- **[Proximal Policy Optimization (PPO)](zenoqx/systems/ppo)**:
  - [PPO](zenoqx/systems/anakin/ff_ppo.py)
  - [PPO with KL penalty](zenoqx/systems/anakin/ff_ppo_penalty.py)
  - [PPO (continuous action space)](zenoqx/systems/anakin/ff_ppo_continuous.py)
  - [Discovered Policy Optimization (continuous action space)](zenoqx/systems/anakin/ff_dpo_continuous.py)
- **[Soft Actor-Critic (SAC)](zenoqx/systems/sac/)**:
  - [SAC](zenoqx/systems/sac/ff_sac.py)

### Planning Methods

- **[Search](zenoqx/systems/search/)**:
  - [AlphaZero](zenoqx/systems/search/ff_az.py)
  - [Sampled AlphaZero](zenoqx/systems/search/ff_sampled_az.py)

## Contributions

Contributions most welcome! Especially looking for guidance from experienced Equinox users to improve the current approach. Down the road, this repo might also serve as an useful reference for similar migration efforts.

## Changelog

- **26.06.2025** - First commit
