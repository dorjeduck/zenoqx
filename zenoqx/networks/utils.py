from typing import Callable, Dict

import chex
import equinox as eqx

import jax

import jax.numpy as jnp
from typing import Any, Callable
import equinox as eqx
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


'''def parse_rnn_cell(rnn_cell_name: str) -> nn.RNNCellBase:
    """Get the rnn cell."""
    rnn_cells: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "lstm": nn.LSTMCell,
        "optimised_lstm": nn.OptimizedLSTMCell,
        "gru": nn.GRUCell,
        "mgu": nn.MGUCell,
        "simple": nn.SimpleCell,
    }
    return rnn_cells[rnn_cell_name]
'''


def linear_kernel_init(
    input_dim: int, output_dim: int, kernel_init: Initializer, *, key
) -> eqx.nn.Linear:
    """Creates an eqx.nn.Linear layer with orthogonal weight initialization."""
    lkey, wkey = jax.random.split(key)
    layer = eqx.nn.Linear(input_dim, output_dim, key=lkey)
    weights = kernel_init(wkey, layer.weight.shape, dtype=layer.weight.dtype)
    layer = eqx.tree_at(lambda l: l.weight, layer, weights)
    return layer


class SequenceBatchWrapper(eqx.Module):
    """Wrapper that handles sequence reshaping for models with internal batch processing.

    Converts (batch_size, seq_len, ...) inputs to (batch_size * seq_len, ...) for models
    that expect batched inputs, then reshapes outputs back to sequence structure.
    """

    model: eqx.Module

    def __init__(self, model: eqx.Module):
        self.model = model

    def __call__(self, observation: Any) -> Any:
        """
        Handle reshaping for batched sequences.

        Expected input shape: (batch_size, seq_len, ...)
        Model expects: (batch_size * seq_len, ...)
        """
        # Get original sequence dimensions
        first_field = jax.tree_util.tree_leaves(observation)[0]
        batch_size, seq_len = first_field.shape[:2]

        # Flatten sequence dimension into batch dimension
        obs_flattened = jax.tree_map(
            lambda x: x.reshape(batch_size * seq_len, *x.shape[2:]), observation
        )

        # Apply model (which uses internal vmap for batch processing)
        output = self.model(obs_flattened)

        # If output is a distribution, wrap it to handle sequence reshaping
        if hasattr(output, "log_prob"):
            return SequenceDistribution(output, batch_size, seq_len)

        # For other outputs, restore sequence structure
        return jax.tree_map(lambda x: x.reshape(batch_size, seq_len, *x.shape[1:]), output)

    def sample(self, key: chex.PRNGKey, num_samples: int) -> Any:
        return self.model.sample(key, num_samples=num_samples)


class SequenceDistribution(eqx.Module):
    """Distribution wrapper that handles sequence reshaping for probability operations."""

    distribution: Any
    batch_size: int
    seq_len: int

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Compute log_prob with sequence reshaping."""
        # Flatten sequence: (batch_size, seq_len, ...) -> (batch_size * seq_len, ...)
        value_flattened = value.reshape(self.batch_size * self.seq_len, *value.shape[2:])

        # Compute log_prob on flattened sequence
        log_probs_flat = self.distribution.log_prob(value_flattened)

        # Restore sequence structure: (batch_size * seq_len,) -> (batch_size, seq_len)
        return log_probs_flat.reshape(self.batch_size, self.seq_len)

    def sample(self, key: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Sample with sequence reshaping."""
        samples_flat = self.distribution.sample(key, **kwargs)
        return samples_flat.reshape(self.batch_size, self.seq_len, *samples_flat.shape[1:])

    def __getattr__(self, name: str) -> Any:
        """Delegate other methods to the wrapped distribution."""
        return getattr(self.distribution, name)
