from typing import Optional
import chex
import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_vmap
def sample_per_ensemble_deterministic(pi):
    return pi.sample()


@eqx.filter_vmap
def mode_per_ensemble(pi):
    return pi.mode()


@eqx.filter_vmap
def log_prob_per_ensemble(pi, action):
    return pi.log_prob(action)


@eqx.filter_vmap
def sample_per_ensemble(pi, k):
    return pi.sample(k)


@eqx.filter_vmap
def sample_per_ensemble_multiple(pi, key, num_samples: int):
    sample_keys = jax.random.split(key, num_samples)
    return eqx.filter_vmap(lambda k: pi.sample(k))(sample_keys)


def get_policy_from_model(model, observation):
    """Get policy from model, handling both single policy and tuple outputs."""
    model_output = model(observation)
    if isinstance(model_output, tuple):
        policy = model_output[0]
    else:
        policy = model_output
    return policy


def sample_distributions(dist, batch_dim, key: Optional[chex.PRNGKey]):
    if key is None:
        return sample_per_ensemble_deterministic(dist)

    keys = jax.random.split(key, batch_dim)
    return sample_per_ensemble(dist, keys)


def get_action(model, observation, key: Optional[chex.PRNGKey]):
    """Get the policy and action from the model."""
    policy = get_policy_from_model(model, observation)
    return sample_distributions(policy, observation.agent_view.shape[0], key)


def get_mode_action(model, observation):
    """Get the policy and action from the model."""
    policy = get_policy_from_model(model, observation)
    return mode_per_ensemble(policy)


def get_policy_and_action(model, observation, key):
    """Get the policy and action from the model."""
    policy = get_policy_from_model(model, observation)
    action = sample_distributions(policy, observation.agent_view.shape[0], key)
    return policy, action


def get_log_prob(policy, action) -> chex.Array:
    return log_prob_per_ensemble(policy, action)


def get_policy_action_multiple(
    model, observation, key: chex.PRNGKey, num_samples: int
) -> tuple[chex.Array, chex.Array]:

    policy = get_policy_from_model(model, observation)

    keys = jax.random.split(key, observation.agent_view.shape[0])
    actions = sample_per_ensemble_multiple(
        policy, keys, num_samples
    )  # [batch_size, num_samples, action_dim]

    # Transpose to get [num_samples, batch_size, action_dim]
    actions = jnp.transpose(actions, (1, 0, 2))

    return policy, actions
