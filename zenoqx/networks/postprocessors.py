from functools import partial
from typing import Any, Callable, Sequence

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from distrax import Distribution

# Different to bijectors, postprocessors simply wrap the sample and mode methods of a distribution.


class PostProcessedDistribution(Distribution):
    def __init__(
        self, distribution: Distribution, postprocessor: Callable[[chex.Array], chex.Array]
    ):
        self.distribution = distribution
        self.postprocessor = postprocessor

    def sample(self, seed: chex.PRNGKey, sample_shape: Sequence[int] = ()) -> chex.Array:
        return self.postprocessor(self.distribution.sample(seed=seed, sample_shape=sample_shape))

    def mode(self) -> chex.Array:
        return self.postprocessor(self.distribution.mode())

    def log_prob(self, value: chex.Array) -> chex.Array:
        # WARNING: This is only correct if the postprocessor is the identity function!
        # If your postprocessor changes the support, this is not correct.
        return self.distribution.log_prob(value)

    @property
    def event_shape(self):
        return self.distribution.event_shape

    def _sample_n(self, key, n):
        samples = self.distribution._sample_n(key, n)
        return self.postprocessor(samples)

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.distribution, name)


def rescale_to_spec(inputs: chex.Array, minimum: float, maximum: float) -> chex.Array:
    scale = maximum - minimum
    offset = minimum
    inputs = 0.5 * (inputs + 1.0)  # [0, 1]
    output = inputs * scale + offset  # [minimum, maximum]
    return output


def clip_to_spec(inputs: chex.Array, minimum: float, maximum: float) -> chex.Array:
    return jnp.clip(inputs, minimum, maximum)


def tanh_to_spec(inputs: chex.Array, minimum: float, maximum: float) -> chex.Array:
    scale = maximum - minimum
    offset = minimum
    inputs = jax.nn.tanh(inputs)  # [-1, 1]
    inputs = 0.5 * (inputs + 1.0)  # [0, 1]
    output = inputs * scale + offset  # [minimum, maximum]
    return output


class ScalePostProcessor(eqx.Module):
    minimum: float = eqx.static_field()
    maximum: float = eqx.static_field()
    scale_fn: Callable[[chex.Array, float, float], chex.Array] = eqx.static_field()

    def __call__(self, distribution: Distribution) -> Distribution:
        post_processor = partial(self.scale_fn, minimum=self.minimum, maximum=self.maximum)
        return PostProcessedDistribution(distribution, post_processor)


def min_max_normalize(inputs: chex.Array, epsilon: float = 1e-5) -> chex.Array:
    inputs_min = inputs.min(axis=-1, keepdims=True)
    inputs_max = inputs.max(axis=-1, keepdims=True)
    inputs_scale = inputs_max - inputs_min
    inputs_scale = jnp.where(inputs_scale < epsilon, inputs_scale + epsilon, inputs_scale)
    inputs_normed = (inputs - inputs_min) / (inputs_scale)
    return inputs_normed
