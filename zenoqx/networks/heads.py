from typing import Optional, Sequence, Tuple, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

##from flax.linen.initializers import Initializer, lecun_normal, orthogonal

from zenoqx.networks.distributions import (
    AffineTanhTransformedDistribution,
    ClippedBeta,
    DiscreteValuedTfpDistribution,
)

from zenoqx.networks.utils import linear_kernel_init

"""
from tensorflow_probability.substrates.jax.distributions import (
    Categorical,
    Deterministic,
    Independent,
    MultivariateNormalDiag,
    Normal,
    TransformedDistribution,
)
"""


class CategoricalHead(eqx.Module):
    linear: eqx.nn.Linear
    action_dim: Union[int, Sequence[int]] = eqx.static_field()

    def __init__(self, input_dim, action_dim, *, key):
        self.action_dim = action_dim
        output_dim = int(np.prod(action_dim))

        self.linear = linear_kernel_init(
            input_dim, output_dim, kernel_init=jax.nn.initializers.orthogonal(scale=0.01), key=key
        )

    def __call__(self, embedding: jnp.ndarray) -> distrax.Categorical:
        logits = self.linear(embedding)
        if not isinstance(self.action_dim, int):
            logits = logits.reshape(self.action_dim)
        return distrax.Categorical(logits=logits)


class NormalAffineTanhDistributionHead(eqx.Module):
    linear_loc: eqx.nn.Linear
    linear_scale: eqx.nn.Linear
    action_dim: int = eqx.static_field()
    minimum: float = eqx.static_field()
    maximum: float = eqx.static_field()
    min_scale: float = eqx.static_field()

    def __init__(self, input_dim, action_dim, minimum, maximum, min_scale=1e-3, *, key):
        self.action_dim = action_dim
        self.minimum = minimum
        self.maximum = maximum
        self.min_scale = min_scale
        key1, key2 = jax.random.split(key)

        initializer = jax.nn.initializers.orthogonal(scale=0.01)

        self.linear_loc = linear_kernel_init(input_dim, action_dim, initializer, key=key1)
        self.linear_scale = linear_kernel_init(
            input_dim, action_dim, kernel_init=initializer, key=key2
        )

    def __call__(self, embedding: jnp.ndarray) -> distrax.Independent:
        loc = self.linear_loc(embedding)

        scale = jax.nn.softplus(self.linear_scale(embedding)) + self.min_scale
        distribution = distrax.Normal(loc=loc, scale=scale)

        # Broadcast minimum and maximum to the same shape as loc and scale
        # to ensure correct broadcasting in the affine transformation.
        minimum = jnp.broadcast_to(self.minimum, loc.shape)
        maximum = jnp.broadcast_to(self.maximum, loc.shape)

        # Apply transformation first, then Independent (matching original Flax code)
        transformed = AffineTanhTransformedDistribution(distribution, minimum, maximum)
        return distrax.Independent(transformed, reinterpreted_batch_ndims=1)


"""TODOTODO
class BetaDistributionHead(nn.Module):

    action_dim: int
    minimum: float
    maximum: float
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> Independent:

        # Use alpha and beta >= 1 according to [Chou et. al, 2017]
        alpha = (
            jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)) + 1
        )
        beta = (
            jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)) + 1
        )
        # Calculate scale and shift for the affine transformation to achieve the range
        # [minimum, maximum].
        scale = self.maximum - self.minimum
        shift = self.minimum
        affine_bijector = distrax.Chain([distrax.Shift(shift), distrax.Scale(scale)])

        transformed_distribution = distrax.Transformed(
            ClippedBeta(alpha, beta), affine_bijector
        )

        return distrax.Independent(
            transformed_distribution,
            reinterpreted_batch_ndims=1,
        )


class MultivariateNormalDiagHead(nn.Module):

    action_dim: int
    init_scale: float = 0.3
    min_scale: float = 1e-3
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.DistributionLike:
        loc = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding)
        scale = jax.nn.softplus(nn.Dense(self.action_dim, kernel_init=self.kernel_init)(embedding))
        scale *= self.init_scale / jax.nn.softplus(0.0)
        scale += self.min_scale
        return MultivariateNormalDiag(loc=loc, scale_diag=scale)

"""


class DeterministicHead(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, input_dim: int, action_dim: int, *, key):
        self.linear = linear_kernel_init(
            input_dim, action_dim, kernel_init=jax.nn.initializers.orthogonal(scale=0.01), key=key
        )

    def __call__(self, embedding: jnp.ndarray) -> distrax.Deterministic:
        loc = self.linear(embedding)
        return distrax.Deterministic(loc=loc)


class ScalarCriticHead(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, input_dim, *, key):
        self.linear = linear_kernel_init(
            input_dim, 1, kernel_init=jax.nn.initializers.orthogonal(scale=1.0), key=key
        )

    def __call__(self, embedding: jnp.ndarray) -> jnp.ndarray:
        return jnp.squeeze(self.linear(embedding), axis=-1)


'''
class CategoricalCriticHead(nn.Module):

    num_atoms: int = 601
    vmax: Optional[float] = None
    vmin: Optional[float] = None
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> distrax.DistributionLike:
        vmax = self.vmax if self.vmax is not None else 0.5 * (self.num_atoms - 1)
        vmin = self.vmin if self.vmin is not None else -1.0 * vmax

        output = DiscreteValuedTfpHead(
            vmin=vmin,
            vmax=vmax,
            logits_shape=(),
            num_atoms=self.num_atoms,
            kernel_init=self.kernel_init,
        )(embedding)

        return output


class DiscreteValuedTfpHead(nn.Module):
    """Represents a parameterized discrete valued distribution.

    The returned distribution is essentially a `tfd.Categorical` that knows its
    support and thus can compute the mean value.
    If vmin and vmax have shape S, this will store the category values as a
    Tensor of shape (S*, num_atoms).

    Args:
        vmin: Minimum of the value range
        vmax: Maximum of the value range
        num_atoms: The atom values associated with each bin.
        logits_shape: The shape of the logits, excluding batch and num_atoms
        dimensions.
        kernel_init: The initializer for the dense layer.
    """

    vmin: float
    vmax: float
    num_atoms: int
    logits_shape: Optional[Sequence[int]] = None
    kernel_init: Initializer = lecun_normal()

    def setup(self) -> None:
        self._values = np.linspace(self.vmin, self.vmax, num=self.num_atoms, axis=-1)
        if not self.logits_shape:
            logits_shape: Sequence[int] = ()
        else:
            logits_shape = self.logits_shape
        self._logits_shape = (
            *logits_shape,
            self.num_atoms,
        )
        self._logits_size = np.prod(self._logits_shape)
        self._net = nn.Dense(self._logits_size, kernel_init=self.kernel_init)

    def __call__(self, inputs: chex.Array) -> distrax.DistributionLike:
        logits = self._net(inputs)
        logits = logits.reshape(logits.shape[:-1] + self._logits_shape)
        return DiscreteValuedTfpDistribution(values=self._values, logits=logits)

'''


class DiscreteQNetworkHead(eqx.Module):
    linear: eqx.nn.Linear
    epsilon: float = eqx.static_field()

    def __init__(
        self,
        input_dim,
        action_dim,
        kernel_init=jax.nn.initializers.orthogonal(scale=1.0),
        epsilon=0.1,
        *,
        key=None,
    ):

        if key == None:
            key = jax.random.key(0)

        self.linear = self.linear = linear_kernel_init(
            input_dim, action_dim, kernel_init=kernel_init, key=key
        )
        self.epsilon = epsilon

    def __call__(self, embedding: jnp.ndarray) -> distrax.EpsilonGreedy:
        q_values = self.linear(embedding)
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)


"""TODOTODO
class PolicyValueHead(nn.Module):
    action_head: nn.Module
    critic_head: nn.Module

    @nn.compact
    def __call__(
        self, embedding: chex.Array
    ) -> Tuple[distrax.DistributionLike, Union[chex.Array, distrax.DistributionLike]]:

        action_distribution = self.action_head(embedding)
        value = self.critic_head(embedding)

        return action_distribution, value

"""


class DistributionalDiscreteQNetwork(eqx.Module):
    linear: eqx.nn.Linear
    action_dim: int = eqx.static_field()
    num_atoms: int = eqx.static_field()
    vmin: float = eqx.static_field()
    vmax: float = eqx.static_field()
    epsilon: float = eqx.static_field()

    def __init__(self, input_dim, action_dim, num_atoms, vmin, vmax, epsilon=0.1, *, key):

        self.linear = linear_kernel_init(
            input_dim,
            action_dim * num_atoms,
            kernel_init=jax.nn.initializers.lecun_normal(),
            key=key,
        )
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.epsilon = epsilon

    def __call__(self, embedding: jnp.ndarray):
        atoms = jnp.linspace(self.vmin, self.vmax, self.num_atoms)
        q_logits = self.linear(embedding)
        q_logits = q_logits.reshape(self.action_dim, self.num_atoms)
        q_dist = jax.nn.softmax(q_logits, axis=-1)
        q_values = jnp.sum(q_dist * atoms, axis=1)
        q_values = jax.lax.stop_gradient(q_values)
        atoms_broadcast = atoms  # No batch, so just atoms
        return (
            distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon),
            q_logits,
            atoms_broadcast,
        )


class DistributionalContinuousQNetwork(eqx.Module):
    linear: eqx.nn.Linear
    num_atoms: int = eqx.static_field()
    vmin: float = eqx.static_field()
    vmax: float = eqx.static_field()

    def __init__(self, input_dim, num_atoms, vmin, vmax, *, key):
        self.linear = linear_kernel_init(
            input_dim, num_atoms, kernel_init=jax.nn.initializers.lecun_normal(), key=key
        )
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, embedding: jnp.ndarray):
        atoms = jnp.linspace(self.vmin, self.vmax, self.num_atoms)
        q_logits = self.linear(embedding)
        q_dist = jax.nn.softmax(q_logits)
        q_value = jnp.sum(q_dist * atoms, axis=-1)
        atoms = jnp.broadcast_to(atoms, (*q_value.shape, self.num_atoms))
        return q_value, q_logits, atoms


class QuantileDiscreteQNetwork(eqx.Module):
    linear: eqx.nn.Linear
    action_dim: int = eqx.static_field()
    epsilon: float = eqx.static_field()
    num_quantiles: int = eqx.static_field()

    def __init__(self, input_dim, action_dim, num_quantiles, epsilon=0.1, *, key):

        self.linear = linear_kernel_init(
            input_dim,
            action_dim * num_quantiles,
            kernel_init=jax.nn.initializers.lecun_normal(),
            key=key,
        )

        self.action_dim = action_dim
        self.epsilon = epsilon
        self.num_quantiles = num_quantiles

    def __call__(self, embedding: jnp.ndarray):
        # No batch: embedding is 1D (input_dim,)
        q_logits = self.linear(embedding)
        q_dist = jnp.reshape(q_logits, (self.action_dim, self.num_quantiles))
        q_values = jnp.mean(q_dist, axis=-1)
        q_values = jax.lax.stop_gradient(q_values)
        return distrax.EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_dist


"""
class LinearHead(nn.Module):
    output_dim: int
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:

        return nn.Dense(self.output_dim, kernel_init=self.kernel_init)(embedding)
"""
