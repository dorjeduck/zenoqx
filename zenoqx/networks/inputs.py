import chex
import jax
import jax.numpy as jnp
import equinox as eqx

from zenoqx.base_types import Observation


class EmbeddingInput(eqx.Module):
    """JAX Array Input."""

    def __call__(self, embedding: chex.Array) -> chex.Array:
        return embedding


class ObservationInput(eqx.Module):
    """Only Observation Input."""

    def __call__(self, observation: Observation) -> chex.Array:
        observation = observation.agent_view
        return observation


class ObservationActionInput(eqx.Module):
    """Observation and Action Input."""

    def __call__(self, observation: Observation, action: chex.Array) -> chex.Array:
        observation = observation.agent_view
        x = jnp.concatenate([observation, action], axis=-1)
        return x


class EmbeddingActionInput(eqx.Module):

    action_dim: int

    def __call__(self, observation_embedding: chex.Array, action: chex.Array) -> chex.Array:
        x = jnp.concatenate([observation_embedding, action], axis=-1)
        return x


class EmbeddingActionOnehotInput(eqx.Module):

    action_dim: int

    def __call__(self, observation_embedding: chex.Array, action: chex.Array) -> chex.Array:
        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([observation_embedding, action_one_hot], axis=-1)
        return x
