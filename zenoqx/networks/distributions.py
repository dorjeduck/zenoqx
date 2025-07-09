from typing import Any, Optional, Sequence, Union
import jax
import jax.numpy as jnp
import numpy as np
import chex
import equinox as eqx
from distreqx.distributions import AbstractDistribution, Transformed, Normal, Categorical
from distreqx.bijectors import Chain, Shift, ScalarAffine, Tanh


class AffineTanhTransformedDistribution(Transformed):
    """Distribution followed by tanh and then affine transformations."""

    _min_threshold: float
    _max_threshold: float
    _log_prob_left: chex.Array
    _log_prob_right: chex.Array

    def __init__(
        self,
        distribution: AbstractDistribution,
        minimum: float,
        maximum: float,
        epsilon: float = 1e-3,
    ) -> None:
        """Initialize the distribution with a tanh and affine bijector.

        Args:
          distribution: The distribution to transform.
          minimum: Lower bound of the target range.
          maximum: Upper bound of the target range.
          epsilon: epsilon value for numerical stability.
            epsilon is used to compute the log of the average probability distribution
            outside the clipping range, i.e. on the interval
            [-inf, atanh(inverse_affine(minimum))] for log_prob_left and
            [atanh(inverse_affine(maximum)), inf] for log_prob_right.
        """
        scale = jnp.asarray((maximum - minimum) / 2.0)
        shift = jnp.asarray((minimum + maximum) / 2.0)

        joint_bijector = Chain(
            [
                Shift(shift),
                ScalarAffine(scale=scale, shift=0.0),
                Tanh(),
            ]
        )
        super().__init__(distribution, joint_bijector)

        # Store threshold parameters for clipping
        self._min_threshold = minimum + epsilon
        self._max_threshold = maximum - epsilon

        # Calculate log prob bounds for clipping
        min_inverse_threshold = joint_bijector.inverse(jnp.asarray(self._min_threshold))
        max_inverse_threshold = joint_bijector.inverse(jnp.asarray(self._max_threshold))

        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(epsilon)

        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = self.distribution.log_cdf(min_inverse_threshold) - log_epsilon
        self._log_prob_right = (
            self.distribution.log_survival_function(max_inverse_threshold) - log_epsilon
        )

    def log_prob(self, event: chex.Array) -> chex.Array:
        # Without this clip there would be NaNs in the inner jnp.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, self._min_threshold, self._max_threshold)

        return jnp.where(
            event <= self._min_threshold,
            self._log_prob_left,
            jnp.where(event >= self._max_threshold, self._log_prob_right, super().log_prob(event)),
        )

    def mode(self) -> chex.Array:
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, key: chex.PRNGKey = None) -> chex.Array:
        """Compute entropy of the transformed distribution.

        Since distreqx.bijector.forward_log_det_jacobian doesn't support event_ndims,
        we need to manually handle the case where we want event_ndims=0 behavior
        (all dimensions treated as batch dimensions).

        Args:
            key: Random key for sampling (if needed by underlying distribution)

        Returns:
            Entropy of the transformed distribution with proper shape handling
        """
        base_entropy = self.distribution.entropy()
        sample = self.distribution.sample(key)
        log_det_jacobian = self.bijector.forward_log_det_jacobian(sample)

        # Ensure shapes are compatible for addition
        # For event_ndims=0 behavior, we want log_det_jacobian to have the same shape as base_entropy
        # If base_entropy is scalar and log_det_jacobian has extra dimensions, sum them
        if base_entropy.ndim == 0 and log_det_jacobian.ndim > 0:
            # Sum over all dimensions to get a scalar (event_ndims=0 for scalar base distribution)
            log_det_jacobian = jnp.sum(log_det_jacobian)
        elif base_entropy.ndim > 0 and log_det_jacobian.ndim > base_entropy.ndim:
            # Sum over the extra dimensions to match base_entropy shape
            # This handles the case where log_det_jacobian has more dimensions than expected
            axes_to_sum = tuple(range(base_entropy.ndim, log_det_jacobian.ndim))
            log_det_jacobian = jnp.sum(log_det_jacobian, axis=axes_to_sum)

        return base_entropy + log_det_jacobian

    def kl_divergence(self, other: "AffineTanhTransformedDistribution") -> chex.Array:
        """Compute KL divergence between two AffineTanhTransformedDistribution instances.

        Since both distributions use the same bijector transform, we can compute
        the KL divergence between the underlying base distributions.
        """
        return self.distribution.kl_divergence(other.distribution)


# TODO: Implement Beta in Distreqx and uncomment this
# class ClippedBeta(Beta):
#     """Beta distribution with clipped samples."""
#
#     def sample(
#         self,
#         sample_shape: Sequence[int] = (),
#         seed: Optional[chex.PRNGKey] = None,
#         name: str = "sample",
#         **kwargs: Any,
#     ) -> chex.Array:
#         _epsilon = 1e-7
#         sample = super().sample(sample_shape, seed, **kwargs)
#         clipped_sample = jnp.clip(sample, _epsilon, 1 - _epsilon)
#         return clipped_sample


class DiscreteValuedTfpDistribution(Categorical):
    """This is a generalization of a categorical distribution.

    The support for the DiscreteValued distribution can be any real valued range,
    whereas the categorical distribution has support [0, n_categories - 1] or
    [1, n_categories]. This generalization allows us to take the mean of the
    distribution over its support.
    """

    def __init__(
        self,
        values: chex.Array,
        logits: Optional[chex.Array] = None,
        probs: Optional[chex.Array] = None,
        name: str = "DiscreteValuedDistribution",
    ):
        parameters = dict(locals())
        self._values = np.asarray(values)
        self._logits: Optional[chex.Array] = None
        self._probs: Optional[chex.Array] = None

        if logits is not None:
            logits = jnp.asarray(logits)
            # chex.assert_shape(logits, (..., *self._values.shape))

        if probs is not None:
            probs = jnp.asarray(probs)
            # chex.assert_shape(probs, (..., *self._values.shape))

        super().__init__(logits=logits, probs=probs)
        self._parameters = parameters

    @property
    def values(self) -> chex.Array:
        return self._values

    @property
    def logits(self) -> chex.Array:
        if self._logits is None:
            self._logits = jax.nn.log_softmax(self._probs)
        return self._logits

    @property
    def probs(self) -> chex.Array:
        if self._probs is None:
            self._probs = jax.nn.softmax(self._logits)
        return self._probs

    def _sample_n(self, key: chex.PRNGKey, n: int) -> chex.Array:
        indices = super()._sample_n(key=key, n=n)
        return jnp.take_along_axis(self._values, indices, axis=-1)

    def mean(self) -> chex.Array:
        return jnp.sum(self.probs_parameter() * self._values, axis=-1)

    def variance(self) -> chex.Array:
        dist_squared = jnp.square(jnp.expand_dims(self.mean(), -1) - self._values)
        return jnp.sum(self.probs_parameter() * dist_squared, axis=-1)

    def _event_shape(self) -> chex.Array:
        return jnp.zeros((), dtype=jnp.int32)

    def _event_shape_tensor(self) -> chex.Array:
        return []


class MultiDimActionDistribution(eqx.Module):
    """Distribution for multi-dimensional action spaces.

    Each action dimension is modeled by a separate univariate distribution,
    but they collectively represent a single action. Following Distreqx style:
    - Each dimension's distribution is independent
    - vmap is used to apply operations across dimensions
    - Log probabilities are summed because they represent one action
    - No internal batch handling (for batches of states/observations)

    Args:
        distributions: A sequence of univariate distributions, one per action dimension
    """

    distributions: AbstractDistribution

    def __init__(self, distributions: AbstractDistribution) -> None:
        """Initialize with a sequence of univariate distributions.

        Args:
            distributions: Sequence of distributions, one per action dimension
        """
        self.distributions = distributions

    def sample(
        self, key: Optional[chex.PRNGKey] = None, seed: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        """Sample one action from the multi-dimensional action space.

        Args:
            key: Random key for sampling
            seed: Alternative name for key parameter, for compatibility

        Returns:
            Array with shape (action_dim,)
        """
        k = key if key is not None else seed
        if k is not None:
            # Split the key for each dimension
            keys = jax.random.split(k, len(self.distributions))
            return jnp.array([d.sample(key=sub_k) for d, sub_k in zip(self.distributions, keys)])
        else:
            return jnp.array([d.sample() for d in self.distributions])

    def mode(self) -> chex.Array:
        """Get the mode of the action distribution.

        Returns:
            Array with shape (action_dim,)
        """
        return jnp.array([d.mode() for d in self.distributions])

    def log_prob(self, value: chex.Array) -> chex.Array:
        """Get total log probability of an action.

        Args:
            value: Action to evaluate, shape (action_dim,)

        Returns:
            Total log probability (summed across dimensions)
        """
        per_dim = jnp.array([d.log_prob(v) for d, v in zip(self.distributions, value)])
        return jnp.sum(per_dim)  # Sum because it's one action

    def entropy(self, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """Get total entropy of the action distribution.

        Args:
            key: Random key for sampling entropy estimate

        Returns:
            Total entropy (summed across dimensions)
        """
        if key is not None:
            # Split the key for each dimension
            keys = jax.random.split(key, len(self.distributions))
            per_dim = jnp.array([d.entropy(k) for d, k in zip(self.distributions, keys)])
        else:
            # No key needed, use simple computation
            per_dim = jnp.array([d.entropy() for d in self.distributions])

        return jnp.sum(per_dim)  # Sum because it's one action

    @property
    def batch_shape(self) -> tuple:
        """The shape of the batch of distributions."""
        return self.distributions.shape

    def __getitem__(self, idx) -> AbstractDistribution:
        """Get a single distribution from the batch."""
        return self.distributions[idx]
