from typing import List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, Initializer, PrecisionLike

from zenoqx.networks.utils import parse_activation_fn

default_kernel_init = initializers.lecun_normal()


class StackedRNN(nn.Module):
    """
    A class representing a stacked recurrent neural network (RNN).

    Attributes:
        rnn_size (int): The size of the hidden state for each RNN cell.
        rnn_cls (nn.Module): The class for the RNN cell to be used.
        num_layers (int): The number of RNN layers.
        activation_fn (str): The activation function to use in each RNN cell (default is "tanh").
    """

    rnn_size: int
    rnn_cls: nn.Module
    num_layers: int
    activation_fn: str = "sigmoid"

    def setup(self) -> None:
        """Set up the RNN cells for the stacked RNN."""
        self.cells = [
            self.rnn_cls(
                features=self.rnn_size, activation_fn=parse_activation_fn(self.activation_fn)
            )
            for _ in range(self.num_layers)
        ]

    def __call__(
        self, all_rnn_states: List[chex.ArrayTree], x: chex.Array
    ) -> Tuple[List[chex.ArrayTree], chex.Array]:
        """
        Run the stacked RNN cells on the input.

        Args:
            all_rnn_states (List[chex.ArrayTree]): List of RNN states for each layer.
            x (chex.Array): Input to the RNN.

        Returns:
            Tuple[List[chex.ArrayTree], chex.Array]: A tuple containing the a list of
                the RNN states of each RNN and the output of the last layer.
        """
        # Ensure all_rnn_states is a list
        if not isinstance(all_rnn_states, list):
            all_rnn_states = [all_rnn_states]

        assert (
            len(all_rnn_states) == self.num_layers
        ), f"Expected {self.num_layers} RNN states, but got {len(all_rnn_states)}."

        new_states = []
        for cell, rnn_state in zip(self.cells, all_rnn_states):
            new_rnn_state, x = cell(rnn_state, x)
            new_states.append(new_rnn_state)

        return new_states, x


import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Callable


class NoisyLinear(eqx.Module):
    """
    Noisy Linear Layer using independent Gaussian noise
    as defined in Fortunato et al. (2018):

    y = (μ_w + σ_w * ε_w) . x + μ_b + σ_b * ε_b,

    where μ_w, μ_b, σ_w, σ_b are learnable parameters
    and ε_w, ε_b are noise random variables generated using
    Factorised Gaussian Noise.

    Attributes:
    * features (int): The number of output features.
    * sigma_zero (float): Initialization value for σ terms.
    """

    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    sigma_w: jnp.ndarray
    sigma_b: Optional[jnp.ndarray]
    features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    sigma_zero: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        features: int,
        use_bias: bool = True,
        sigma_zero: float = 0.5,
        key: chex.PRNGKey = None,
        *,
        kernel_init: Optional[Callable] = None,
        bias_init: Optional[Callable] = None,
    ):
        assert key is not None
        self.features = features
        self.use_bias = use_bias
        self.sigma_zero = sigma_zero
        kernel_init = kernel_init or jax.nn.initializers.lecun_normal()
        bias_init = bias_init or jax.nn.initializers.zeros
        k1, k2 = jax.random.split(key)
        self.weight = kernel_init(k1, (input_dim, features))
        sigma_init = sigma_zero / jnp.sqrt(input_dim)
        self.sigma_w = jnp.ones((input_dim, features)) * sigma_init
        if use_bias:
            self.bias = bias_init(k2, (features,))
            self.sigma_b = jnp.ones((features,)) * sigma_init
        else:
            self.bias = None
            self.sigma_b = None

    def _scale_noise(self, x):
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))

    def _get_noise_matrix_and_vect(self, input_dim, key):
        k1, k2 = jax.random.split(key)
        row_noise = self._scale_noise(jax.random.normal(k1, (input_dim,)))
        col_noise = self._scale_noise(jax.random.normal(k2, (self.features,)))
        noise_matrix = jnp.outer(row_noise, col_noise)
        return noise_matrix, col_noise

    def __call__(self, inputs: jnp.ndarray, key: chex.PRNGKey) -> jnp.ndarray:
        input_dim = inputs.shape[-1]
        eps_w, eps_b = self._get_noise_matrix_and_vect(input_dim, key)
        noisy_weight = self.weight + self.sigma_w * eps_w
        y = jnp.dot(inputs, noisy_weight)
        if self.use_bias and self.bias is not None and self.sigma_b is not None:
            noisy_bias = self.bias + self.sigma_b * eps_b
            y = y + noisy_bias
        return y
