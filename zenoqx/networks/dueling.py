from typing import Sequence

import chex
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from zenoqx.distreqx.distributions import EpsilonGreedy
from zenoqx.networks.layers import NoisyLinear
from zenoqx.networks.torso import MLPTorso, NoisyMLPTorso

"""
class DuelingQNetwork(nn.Module):

    action_dim: int
    epsilon: float
    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:

        value = MLPTorso(
            (*self.layer_sizes, 1),
            self.activation,
            self.use_layer_norm,
            self.kernel_init,
            activate_final=False,
        )(inputs)
        advantages = MLPTorso(
            (*self.layer_sizes, self.action_dim),
            self.activation,
            self.use_layer_norm,
            self.kernel_init,
            activate_final=False,
        )(inputs)

        # Advantages have zero mean.
        advantages -= jnp.mean(advantages, axis=-1, keepdims=True)

        q_values = value + advantages

        return EpsilonGreedy(preferences=q_values, epsilon=self.epsilon)


class DistributionalDuelingQNetwork(nn.Module):
    num_atoms: int
    vmax: float
    vmin: float
    action_dim: int
    epsilon: float
    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))

    @nn.compact
    def __call__(self, inputs: chex.Array) -> chex.Array:

        value_torso = MLPTorso(
            self.layer_sizes, self.activation, self.use_layer_norm, self.kernel_init
        )(inputs)
        advantages_torso = MLPTorso(
            self.layer_sizes,
            self.activation,
            self.use_layer_norm,
            self.kernel_init,
        )(inputs)

        value_logits = nn.Dense(self.num_atoms, kernel_init=self.kernel_init)(value_torso)
        value_logits = jnp.reshape(value_logits, (-1, 1, self.num_atoms))
        adv_logits = nn.Dense(self.action_dim * self.num_atoms, kernel_init=self.kernel_init)(
            advantages_torso
        )
        adv_logits = jnp.reshape(adv_logits, (-1, self.action_dim, self.num_atoms))
        q_logits = value_logits + adv_logits - adv_logits.mean(axis=1, keepdims=True)

        atoms = jnp.linspace(self.vmin, self.vmax, self.num_atoms)
        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * atoms, axis=2)
        q_values = jax.lax.stop_gradient(q_values)
        atoms = jnp.broadcast_to(atoms, (q_values.shape[0], self.num_atoms))
        return EpsilonGreedy(preferences=q_values, epsilon=self.epsilon), q_logits, atoms

"""


class NoisyDistributionalDuelingQNetwork(eqx.Module):
    value_torso: eqx.Module
    adv_torso: eqx.Module
    value_linear: eqx.Module
    adv_linear: eqx.Module
    num_atoms: int = eqx.static_field()
    vmax: float = eqx.static_field()
    vmin: float = eqx.static_field()
    action_dim: int = eqx.static_field()
    epsilon: float = eqx.static_field()
    eval_epsilon: float = eqx.static_field()
    sigma_zero: float = eqx.static_field()

    def __init__(
        self,
        input_dim: int,
        layer_sizes: Sequence[int],
        action_dim: int,
        num_atoms: int,
        vmin: float,
        vmax: float,
        epsilon: float = 0.1,
        eval_epsilon: float = 0.05,
        sigma_zero: float = 0.5,
        activation: str = "relu",
        use_layer_norm: bool = False,
        *,
        key,
    ):
        keys = jax.random.split(key, 4)
        self.value_torso = NoisyMLPTorso(
            input_dim=input_dim,
            layer_sizes=layer_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm,
            sigma_zero=sigma_zero,
            key=keys[0],
        )
        self.adv_torso = NoisyMLPTorso(
            input_dim=input_dim,
            layer_sizes=layer_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm,
            sigma_zero=sigma_zero,
            key=keys[1],
        )
        last_layer_dim = layer_sizes[-1] if layer_sizes else input_dim
        self.value_linear = NoisyLinear(
            input_dim=last_layer_dim,
            features=num_atoms,
            sigma_zero=sigma_zero,
            key=keys[2],
        )
        self.adv_linear = NoisyLinear(
            input_dim=last_layer_dim,
            features=action_dim * num_atoms,
            sigma_zero=sigma_zero,
            key=keys[3],
        )
        self.num_atoms = num_atoms
        self.vmax = vmax
        self.vmin = vmin
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.eval_epsilon = eval_epsilon
        self.sigma_zero = sigma_zero

    def __call__(
        self, embedding: jnp.ndarray, key: chex.PRNGKey, *, inference: bool = False
    ) -> distrax.DistributionLike:

        if inference:
            eps = self.eval_epsilon
        else:
            eps = self.epsilon

        keys = jax.random.split(key, 4)

        value_torso = self.value_torso(embedding, key=keys[0])

        adv_torso = self.adv_torso(embedding, key=keys[1])
        value_logits = eqx.filter_vmap(self.value_linear)(value_torso, keys[2])
        value_logits = value_logits.reshape(1, self.num_atoms)
        adv_logits = eqx.filter_vmap(self.adv_linear)(adv_torso, key=keys[3])
        adv_logits = adv_logits.reshape(self.action_dim, self.num_atoms)
        q_logits = value_logits + adv_logits - adv_logits.mean(axis=0, keepdims=True)
        atoms = jnp.linspace(self.vmin, self.vmax, self.num_atoms)
        q_dist = jax.nn.softmax(q_logits, axis=-1)
        q_values = jnp.sum(q_dist * atoms, axis=1)
        q_values = jax.lax.stop_gradient(q_values)
        atoms = atoms
        return EpsilonGreedy(preferences=q_values, epsilon=eps), q_logits, atoms
