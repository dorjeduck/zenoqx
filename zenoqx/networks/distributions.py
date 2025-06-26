from typing import Any, Optional, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
import distrax


class AffineTanhTransformedDistribution(distrax.Transformed):
    """Distribution followed by tanh and then affine transformations.

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

    def __init__(
        self,
        distribution: distrax.Distribution,
        minimum: float,
        maximum: float,
        epsilon: float = 1e-3,
    ) -> None:
        # Calculate scale and shift for the affine transformation to achieve the range [minimum, maximum] after tanh
        scale = (maximum - minimum) / 2.0
        shift = (minimum + maximum) / 2.0

        joint_bijector = distrax.Chain(
            [distrax.ScalarAffine(scale=scale, shift=shift), distrax.Tanh()]
        )
        super().__init__(distribution, joint_bijector)

        self._min_threshold = minimum + epsilon
        self._max_threshold = maximum - epsilon
        min_inverse_threshold = self.bijector.inverse(jnp.array(self._min_threshold))
        max_inverse_threshold = self.bijector.inverse(jnp.array(self._max_threshold))
        log_epsilon = jnp.log(epsilon)
        self._log_prob_left = self.distribution.log_cdf(min_inverse_threshold) - log_epsilon
        self._log_prob_right = (
            self.distribution.log_survival_function(max_inverse_threshold) - log_epsilon
        )

    def log_prob(self, event: chex.Array) -> chex.Array:
        # Without this clip there would be NaNs in the inner tf.where and that causes issues for some reasons.
        event = jnp.clip(event, self._min_threshold, self._max_threshold)
        return jnp.where(
            event <= self._min_threshold,
            self._log_prob_left,
            jnp.where(event >= self._max_threshold, self._log_prob_right, super().log_prob(event)),
        )

    def mode(self) -> chex.Array:
        return self.bijector.forward(self.distribution.mode())

    # def entropy(self, seed: chex.PRNGKey = None) -> chex.Array:
    #    # https://github.com/google-deepmind/distrax/blob/7fc5bd7efff4a7144d175199159f115c3e68a3cf/distrax/_src/bijectors/tfp_compatible_bijector.py#L145
    #
    #    return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
    #        self.distribution.sample(seed=seed),  ## TODO event_ndims=0
    #    )

    def entropy(self, seed: chex.PRNGKey = None) -> chex.Array:
        # https://github.com/google-deepmind/distrax/blob/7fc5bd7efff4a7144d175199159f115c3e68a3cf/distrax/_src/bijectors/tfp_compatible_bijector.py#L145
        # The entropy of a transformed distribution is H(Y) = H(X) + E_X[log|det J(X)|].
        # We approximate the expectation over X with a single sample.
        sample = self.distribution.sample(seed=seed)
        fldj = self.bijector.forward_log_det_jacobian(sample)

        # The log-determinant of the Jacobian must be summed over the event dimensions.
        event_ndims = len(self.event_shape)
        reduce_axis = tuple(range(-event_ndims, 0))

        return self.distribution.entropy() + jnp.sum(fldj, axis=reduce_axis)

    """
    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes: Any = None) -> Any:
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties
    """


class ClippedBeta(distrax.Beta):
    """Beta distribution with clipped samples."""

    def sample(
        self,
        sample_shape: Sequence[int] = (),
        seed: Optional[jax.random.PRNGKey] = None,
        name: str = "sample",
        **kwargs: Any,
    ) -> chex.Array:
        _epsilon = 1e-7
        sample = super().sample(sample_shape, seed, **kwargs)
        clipped_sample = jnp.clip(sample, _epsilon, 1 - _epsilon)
        return clipped_sample


class DiscreteValuedTfpDistribution(distrax.Categorical):
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
