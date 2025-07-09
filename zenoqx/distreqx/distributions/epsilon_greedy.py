"""Epsilon greedy distribution wrapper."""

import distrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class EpsilonGreedy(eqx.Module):
    """Epsilon greedy distribution wrapper around distrax.EpsilonGreedy.

    This wrapper modifies the sample interface to take key as a positional argument
    rather than as a keyword argument (seed=key) as in distrax.
    """

    _preferences: Array = eqx.field()
    _epsilon: float = eqx.field()

    def __init__(self, preferences: Array, epsilon: float):
        """Initialize the epsilon greedy distribution.

        Args:
            preferences: Array of shape (..., n) containing the preferences for each action
            epsilon: Probability of choosing a random action
        """
        self._preferences = preferences
        self._epsilon = epsilon

    @property
    def preferences(self) -> Array:
        """Get the underlying preferences."""
        return self._preferences

    @property
    def epsilon(self) -> float:
        """Get the epsilon value."""
        return self._epsilon

    @property
    def event_shape(self) -> tuple:
        """Get the event shape."""
        return ()

    def sample(self, key: PRNGKeyArray) -> Array:
        """Sample from the distribution using the provided key.

        Args:
            key: A PRNG key used as the random key.

        Returns:
            A sample from the distribution.
        """
        # Create distrax distribution on-the-fly to avoid PyTree issues
        dist = distrax.EpsilonGreedy(preferences=self._preferences, epsilon=self._epsilon)
        return dist.sample(seed=key)

    def log_prob(self, value: Array) -> Array:
        """Compute log probability of value.

        Args:
            value: Value to compute log probability for.

        Returns:
            Log probability of the value.
        """
        # Create distrax distribution on-the-fly to avoid PyTree issues
        dist = distrax.EpsilonGreedy(preferences=self._preferences, epsilon=self._epsilon)
        return dist.log_prob(value)
