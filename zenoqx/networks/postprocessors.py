from functools import partial
from typing import Any, Callable, Optional, Sequence

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from distreqx.distributions import AbstractDistribution

# Different to bijectors, postprocessors simply wrap the sample and mode methods of a distribution.


class PostProcessedDistribution(eqx.Module):
    distribution: eqx.Module
    postprocessor: eqx.Module
    def __init__(
        self, distribution: AbstractDistribution, postprocessor: Callable[[chex.Array], chex.Array]
    ):
        self.distribution = distribution
        self.postprocessor = postprocessor

    def sample(self, key: Optional[chex.PRNGKey] = None, sample_shape: Sequence[int] = ()) -> chex.Array:
      
        return self.postprocessor(self.distribution.sample(key))

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


class ScalePostProcessorFn(eqx.Module):
    minimum: float = eqx.static_field()
    maximum: float = eqx.static_field()
    scale_fn: Callable = eqx.static_field()
    
    def __init__(self, minimum: float, maximum: float, scale_fn: Callable[[chex.Array, float, float], chex.Array]):
        self.minimum = minimum
        self.maximum = maximum
        self.scale_fn = scale_fn
    
    def __call__(self, x: chex.Array) -> chex.Array:
        return self.scale_fn(x, self.minimum, self.maximum)

class ScalePostProcessor(eqx.Module):
    post_processor: ScalePostProcessorFn
    
    def __init__(self, minimum: float, maximum: float, scale_fn: Callable[[chex.Array, float, float], chex.Array]):
        self.post_processor = ScalePostProcessorFn(
            minimum=minimum, 
            maximum=maximum, 
            scale_fn=scale_fn
        )
    
    def __call__(self, distribution: AbstractDistribution) -> AbstractDistribution:
        return PostProcessedDistribution(distribution, self.post_processor)
    
'''
class ScalePostProcessor(eqx.Module):
    minimum: float = eqx.static_field()
    maximum: float = eqx.static_field()
    scale_fn: Callable = eqx.static_field()

    def __init__(self, minimum: float, maximum: float, scale_fn: Callable[[chex.Array, float, float], chex.Array]):
        self.minimum = minimum
        self.maximum = maximum
        self.scale_fn = scale_fn

    def __call__(self, distribution: AbstractDistribution) -> AbstractDistribution:
        def post_process(x: chex.Array) -> chex.Array:
            return self.scale_fn(x, self.minimum, self.maximum)
        return PostProcessedDistribution(distribution, post_process)

'''
def min_max_normalize(inputs: chex.Array, epsilon: float = 1e-5) -> chex.Array:
    inputs_min = inputs.min(axis=-1, keepdims=True)
    inputs_max = inputs.max(axis=-1, keepdims=True)
    inputs_scale = inputs_max - inputs_min
    inputs_scale = jnp.where(inputs_scale < epsilon, inputs_scale + epsilon, inputs_scale)
    inputs_normed = (inputs - inputs_min) / (inputs_scale)
    return inputs_normed
