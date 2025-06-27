# zenoqx

Reinforcement learning with [JAX](https://jax.readthedocs.io/)/[Equinox](https://github.com/patrick-kidger/equinox). Currently focusing on porting [Stoix](https://github.com/EdanToledo/Stoix) from [Flax/Linen](https://flax-linen.readthedocs.io/en/latest/) to Equinox as a starting point.

## Purpose

Personal learning project exploring RL/JAX/Equinox. This evaluates [Equinox](https://github.com/patrick-kidger/equinox) as an alternative to the still widely used [Flax Linen](https://flax-linen.readthedocs.io/en/latest/), whose successor [NNX](https://flax.readthedocs.io/en/latest/nnx/index.html) hasn't clicked for me yet.

Beyond learning, this implementation hopefully will serve as a foundation for experimenting with novel RL approaches once it matures.

## Status

**Alpha** - Experimental, buggy, not production ready.

## Ported Algorithms (Work in Progress)

### Value-Based Methods

- **[Q-Learning](zenoqx/systems/q_learning/)**: [C51](zenoqx/systems/q_learning/ff_c51.py), [DQN](zenoqx/systems/q_learning/ff_dqn.py), [DQN with regularization](zenoqx/systems/q_learning/ff_dqn_reg.py), [Double DQN](zenoqx/systems/q_learning/ff_ddqn.py), [Munchausen DQN](zenoqx/systems/q_learning/ff_mdqn.py), [QR-DQN](zenoqx/systems/q_learning/ff_qr_dqn.py), [Rainbow](zenoqx/systems/q_learning/ff_rainbow.py)

### Policy-Based Methods

- **[Policy Gradient](zenoqx/systems/vpg/)**: [REINFORCE](zenoqx/systems/vpg/ff_reinforce.py)
- **[MPO](zenoqx/systems/mpo/)**: [MPO](zenoqx/systems/mpo/ff_mpo.py), [V-MPO](zenoqx/systems/mpo/ff_vmpo.py)

### Actor-Critic Methods

- **[AWR](zenoqx/systems/awr/)**: [Advantage-Weighted Regression](zenoqx/systems/awr/ff_awr.py)
- **[DDPG](zenoqx/systems/ddpg/)**: [D4PG](zenoqx/systems/ddpg/ff_d4pg.py), [DDPG](zenoqx/systems/ddpg/ff_ddpg.py)
- **[SAC](zenoqx/systems/sac/)**: [Soft Actor-Critic](zenoqx/systems/sac/ff_sac.py)

### Planning Methods

- **[Search](zenoqx/systems/search/)**: [AlphaZero](zenoqx/systems/search/ff_az.py), [Sampled AlphaZero](zenoqx/systems/search/ff_sampled_az.py)

## Contributions

Contributions most welcome! Especially looking for guidance from experienced Equinox users to improve the current approach. Down the road, this repo might also serve as an useful reference for similar migration efforts.

## Changelog

- **26.06.2025** - First commit
