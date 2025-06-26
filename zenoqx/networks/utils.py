from typing import Callable, Dict

import chex
import equinox as eqx
from flax import linen as nn
import jax

from jax.nn.initializers import Initializer


def parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "silu": jax.nn.silu,
        "elu": jax.nn.elu,
        "gelu": jax.nn.gelu,
        "sigmoid": jax.nn.sigmoid,
        "softplus": jax.nn.softplus,
        "swish": jax.nn.swish,
        "identity": lambda x: x,
        "none": lambda x: x,
        "normalise": jax.nn.standardize,
        "softmax": jax.nn.softmax,
        "log_softmax": jax.nn.log_softmax,
        "log_sigmoid": jax.nn.log_sigmoid,
    }
    return activation_fns[activation_fn_name]


def parse_rnn_cell(rnn_cell_name: str) -> nn.RNNCellBase:
    """Get the rnn cell."""
    rnn_cells: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "lstm": nn.LSTMCell,
        "optimised_lstm": nn.OptimizedLSTMCell,
        "gru": nn.GRUCell,
        "mgu": nn.MGUCell,
        "simple": nn.SimpleCell,
    }
    return rnn_cells[rnn_cell_name]


def linear_kernel_init(
    input_dim: int, output_dim: int, kernel_init: Initializer, *, key
) -> eqx.nn.Linear:
    """Creates an eqx.nn.Linear layer with orthogonal weight initialization."""
    lkey, wkey = jax.random.split(key)
    layer = eqx.nn.Linear(input_dim, output_dim, key=lkey)
    weights = kernel_init(
        wkey, layer.weight.shape, dtype=layer.weight.dtype
    )
    layer = eqx.tree_at(lambda l: l.weight, layer, weights)
    return layer