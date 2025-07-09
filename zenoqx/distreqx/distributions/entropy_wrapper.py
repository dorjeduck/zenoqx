"""Wrapper to add sampled entropy functionality to DistributionEnsemble."""

import chex
import equinox as eqx
import jax
from jaxtyping import Array
from typing import Any


def add_sampled_entropy(cls):
    """Decorator to add sampled_entropy method to DistributionEnsemble."""

    def sampled_entropy(self, key: chex.PRNGKey) -> Array:
        """Compute the entropy of each distribution in the ensemble using keys.

        This method handles the case where distributions are wrapped in Independent
        and need to access the underlying distribution's entropy method with a key.
        """
        keys = jax.random.split(key, self.batch_dim)
        return eqx.filter_vmap(lambda dist, key: dist.distribution.entropy(key=key))(
            self.distributions, keys
        )

    # Add the method to the class
    cls.sampled_entropy = sampled_entropy
    return cls


# Alternative: Create a mixin class
class SampledEntropyMixin:
    """Mixin to add sampled entropy functionality."""

    def sampled_entropy(self, key: chex.PRNGKey) -> Array:
        """Compute the entropy of each distribution in the ensemble using keys.

        This method handles the case where distributions are wrapped in Independent
        and need to access the underlying distribution's entropy method with a key.
        """
        keys = jax.random.split(key, self.batch_dim)
        return eqx.filter_vmap(lambda dist, key: dist.distribution.entropy(key=key))(
            self.distributions, keys
        )
