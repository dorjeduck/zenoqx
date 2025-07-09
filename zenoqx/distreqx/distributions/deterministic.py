from typing import Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from ._distribution import (
    AbstractSampleLogProbDistribution,
    AbstractProbDistribution,
    AbstractCDFDistribution,
    AbstractSTDDistribution,
    AbstractSurivialDistribution,
)


class Deterministic(
    AbstractSampleLogProbDistribution,
    AbstractProbDistribution,
    AbstractCDFDistribution,
    AbstractSTDDistribution,
    AbstractSurivialDistribution,
    strict=True,
):
    """Distribution that produces the same value with probability 1."""

    loc: PyTree[Array]

    def __init__(self, loc: PyTree[Array]):
        """**Arguments:**

        - `loc`: The value returned by `sample` and `mean`.
        """
        self.loc = loc

    def sample(self, key: PRNGKeyArray) -> PyTree[Array]:
        """Samples from the distribution (returns loc)."""
        del key  # Unused
        return self.loc

    def log_prob(self, value: PyTree[Array]) -> PyTree[Array]:
        """Returns 0 if value equals loc, -inf otherwise."""
        return jnp.where(jnp.all(value == self.loc), 0.0, -jnp.inf)

    def entropy(self) -> PyTree[Array]:
        """Returns 0 as there is no uncertainty."""
        return jnp.zeros(())

    def mean(self) -> PyTree[Array]:
        """Returns loc."""
        return self.loc

    def median(self) -> PyTree[Array]:
        """Returns loc."""
        return self.loc

    def mode(self) -> PyTree[Array]:
        """Returns loc."""
        return self.loc

    def variance(self) -> PyTree[Array]:
        """Returns 0."""
        return jnp.zeros_like(self.loc)

    def log_cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        """Returns 0 if value >= loc, -inf otherwise."""
        return jnp.where(jnp.all(value >= self.loc), 0.0, -jnp.inf)

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Calculates KL divergence with another Deterministic distribution."""
        if not isinstance(other_dist, Deterministic):
            raise NotImplementedError(
                f"KL divergence between {self.name} and {type(other_dist).__name__} "
                "is not implemented"
            )
        return jnp.where(jnp.all(self.loc == other_dist.loc), 0.0, jnp.inf)

    @property
    def event_shape(self) -> tuple:
        """Returns the event shape."""
        return jnp.shape(self.loc)
