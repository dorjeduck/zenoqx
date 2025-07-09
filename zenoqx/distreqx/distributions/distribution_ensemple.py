import chex
from distreqx import distributions
from typing import Any
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class DistributionEnsemble(distributions.AbstractDistribution):
    """A module that wraps multiple distributions into an ensemble."""

    batch_dim: int = eqx.field(static=True)
    distributions: Any

    def __init__(self, distributions, batch_dim):
        self.distributions = distributions
        self.batch_dim = batch_dim

    def sample(self, key: chex.PRNGKey, num_samples: int = 1) -> Array:
        if num_samples == 1:
            keys = jax.random.split(key, self.batch_dim)
            return eqx.filter_vmap(lambda dist, k: dist.sample(k))(self.distributions, keys)
        else:
            keys = jax.random.split(key, num_samples)

            def single_sample(dist_key):
                dist_keys = jax.random.split(dist_key, self.batch_dim)
                return eqx.filter_vmap(lambda dist, k: dist.sample(k))(
                    self.distributions, dist_keys
                )

            return eqx.filter_vmap(single_sample)(keys)

    def sample_and_log_prob(self, key: chex.PRNGKey, num_samples: int = 1) -> tuple[Array, Array]:
        """Sample and compute log probability."""
        if num_samples == 1:
            # Single sample case
            keys = jax.random.split(key, self.batch_dim)
            samples_and_log_probs = eqx.filter_vmap(lambda dist, k: dist.sample_and_log_prob(k))(
                self.distributions, keys
            )
            samples, log_probs = samples_and_log_probs
            return samples, log_probs
        else:
            # Multiple samples case
            samples = self.sample(key, num_samples)
            # Compute log_prob for each sample
            log_probs = jax.vmap(
                lambda sample_batch: eqx.filter_vmap(lambda dist, s: dist.log_prob(s))(
                    self.distributions, sample_batch
                )
            )(samples)

            return samples, log_probs

    def log_prob(self, value):
        return eqx.filter_vmap(lambda dist, v: dist.log_prob(v))(self.distributions, value)

    def prob(self, value: Array) -> Array:
        """Compute the probability of each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist, v: dist.prob(v))(self.distributions, value)

    @property
    def event_shape(self) -> Any:
        """Get the event shape from the first distribution."""
        # Assuming all distributions in the ensemble have the same event shape
        return (
            self.distributions[0].event_shape
            if hasattr(self.distributions, "__getitem__")
            else self.distributions.event_shape
        )

    @property
    def logits(self) -> Array:
        return self.distributions.logits

    def mode(self):
        return eqx.filter_vmap(lambda dist: dist.mode())(self.distributions)

    def entropy(self) -> Array:
        """Compute the entropy of each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist: dist.entropy())(self.distributions)

    # special use case as workaround for now
    def sampled_entropy(self, key: chex.PRNGKey) -> Array:
        """Compute the entropy of each distribution in the ensemble."""

        keys = jax.random.split(key, self.batch_dim)
        return eqx.filter_vmap(lambda dist, key: dist.distribution.entropy(key))(
            self.distributions, keys
        )

    def mean(self) -> Array:
        """Compute the mean of each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist: dist.mean())(self.distributions)

    def median(self) -> Array:
        """Compute the median of each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist: dist.median())(self.distributions)

    def variance(self) -> Array:
        """Compute the variance of each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist: dist.variance())(self.distributions)

    def stddev(self) -> Array:
        """Compute the standard deviation of each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist: dist.stddev())(self.distributions)

    def cdf(self, value: Array) -> Array:
        """Compute the cumulative distribution function for each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist, v: dist.cdf(v))(self.distributions, value)

    def log_cdf(self, value: Array) -> Array:
        """Compute the log cumulative distribution function for each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist, v: dist.log_cdf(v))(self.distributions, value)

    def survival_function(self, value: Array) -> Array:
        """Compute the survival function for each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist, v: dist.survival_function(v))(self.distributions, value)

    def log_survival_function(self, value: Array) -> Array:
        """Compute the log survival function for each distribution in the ensemble."""
        return eqx.filter_vmap(lambda dist, v: dist.log_survival_function(v))(
            self.distributions, value
        )

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        """Compute the KL divergence between this ensemble and another distribution ensemble."""
        if not isinstance(other_dist, DistributionEnsemble):
            raise ValueError("KL divergence requires another DistributionEnsemble")
        return eqx.filter_vmap(lambda dist1, dist2: dist1.kl_divergence(dist2, **kwargs))(
            self.distributions, other_dist.distributions
        )

    def __call__(self, key: chex.PRNGKey) -> Any:
        """Sample from the ensemble."""
        return self.sample(key)
