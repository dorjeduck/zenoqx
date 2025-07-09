import chex
import equinox as eqx
import jax

from jax import numpy as jnp
from typing import Callable, Sequence

from zenoqx.networks.layers import NoisyLinear
from zenoqx.networks.utils import parse_activation_fn


class MLPTorso(eqx.Module):
    """MLP torso."""

    layers: tuple
    activation_fn: callable = eqx.static_field()
    output_dim: int = eqx.static_field()
    activate_final: bool = eqx.static_field()

    def __init__(
        self,
        input_dim: int,
        layer_sizes,
        activation="relu",
        use_layer_norm=False,
        activate_final=True,
        *,
        key=None,
    ):
        activation_fn = parse_activation_fn(activation)

        if key == None:
            key = jax.random.key(0)
        keys = jax.random.split(key, len(layer_sizes))

        layers = []
        prev_size = input_dim

        for layer_size, subkey in zip(layer_sizes, keys):
            layers.append(
                eqx.nn.Linear(prev_size, layer_size, use_bias=not use_layer_norm, key=subkey)
            )
            if use_layer_norm:
                layers.append(eqx.nn.LayerNorm(layer_size))
            prev_size = layer_size

        self.layers = tuple(layers)
        self.activation_fn = activation_fn
        self.output_dim = layer_sizes[-1] if layer_sizes else input_dim
        self.activate_final = activate_final

    def __call__(self, observation, *, inference=False):

        x = observation
        for i, layer in enumerate(self.layers):
            x = jax.vmap(layer)(x)
            # Apply activation after each layer except possibly the last
            if self.activate_final or i < len(self.layers) - 1:
                x = jax.vmap(self.activation_fn)(x)
        return x


class NoisyMLPTorso(eqx.Module):
    """MLP torso using NoisyLinear layers instead of standard Dense layers."""

    activation_fn: callable = eqx.static_field()
    layer_sizes: Sequence[int] = eqx.static_field()
    use_layer_norm: bool = eqx.static_field()
    kernel_init: Callable = eqx.static_field()
    activate_final: bool = eqx.static_field()
    sigma_zero: float = eqx.static_field()
    layers: list
    layer_norms: list
    output_dim: int = eqx.static_field()

    def __init__(
        self,
        input_dim,
        layer_sizes,
        activation="relu",
        use_layer_norm=False,
        kernel_init=None,
        activate_final=True,
        sigma_zero=0.5,
        *,
        key,
    ):

        self.activation_fn = parse_activation_fn(activation)

        self.layer_sizes = layer_sizes

        self.use_layer_norm = use_layer_norm
        self.kernel_init = (
            kernel_init
            if kernel_init is not None
            else lambda key, shape: jax.nn.initializers.orthogonal(jnp.sqrt(2.0))(key, shape)
        )
        self.activate_final = activate_final
        self.sigma_zero = sigma_zero
        keys = jax.random.split(key, len(layer_sizes))
        self.output_dim = layer_sizes[-1] if layer_sizes else input_dim

        prev_size = input_dim
        self.layers = []
        self.layer_norms = []
        for layer_size, subkey in zip(layer_sizes, keys):
            self.layers.append(
                NoisyLinear(
                    input_dim=prev_size,
                    features=layer_size,
                    use_bias=not self.use_layer_norm,
                    sigma_zero=self.sigma_zero,
                    key=subkey,
                )
            )

            if use_layer_norm:
                self.layer_norms.append(eqx.nn.LayerNorm(layer_size))
            else:
                self.layer_norms.append(None)
            prev_size = layer_size

    def __call__(
        self, observation: jnp.ndarray, *, key: chex.PRNGKey = None, inference: bool = False
    ) -> jnp.ndarray:
        x = observation

        keys = (
            jax.random.split(key, observation.shape[0])
            if key is not None
            else jax.random.key(0)
        )
        for i, layer in enumerate(self.layers):
            x = jax.vmap(layer)(x, keys)
            if self.use_layer_norm and self.layer_norms[i] is not None:
                x = jax.vmap(self.layer_norms[i])(x)
            if self.activate_final or i != len(self.layers) - 1:
                x = jax.vmap(self.activation_fn)(x)

        return x


'''
class CNNTorso(nn.Module):
    """2D CNN torso. Expects input of shape (batch, height, width, channels).
    After this torso, the output is flattened and put through an MLP of
    hidden_sizes."""

    channel_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    channel_first: bool = False
    hidden_sizes: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation

        # If there is a batch of sequences of images
        if observation.ndim > 4:
            return nn.batch_apply.BatchApply(self.__call__)(observation)

        # If the input is in the form of [B, C, H, W], we need to transpose it to [B, H, W, C]
        if self.channel_first:
            x = x.transpose((0, 2, 3, 1))

        # Convolutional layers
        for channel, kernel, stride in zip(self.channel_sizes, self.kernel_sizes, self.strides):
            x = nn.Conv(
                channel, (kernel, kernel), (stride, stride), use_bias=not self.use_layer_norm
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(reduction_axes=(-3, -2, -1))(x)
            x = parse_activation_fn(self.activation)(x)

        # Flatten
        x = x.reshape(*observation.shape[:-3], -1)

        # MLP layers
        x = MLPTorso(
            layer_sizes=self.hidden_sizes,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            kernel_init=self.kernel_init,
            activate_final=True,
        )(x)

        return x
'''
