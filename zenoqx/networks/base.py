from typing import Any, Dict, List, Sequence, Tuple, Union

import chex
import distrax
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp

from zenoqx.base_types import Observation, RNNObservation
from zenoqx.networks.inputs import ObservationInput
from zenoqx.networks.utils import parse_rnn_cell

from typing import Sequence, Union, Tuple


class FeedForwardActor(eqx.Module):
    """Simple Feedforward Actor Network."""

    action_head: eqx.Module
    torso: eqx.Module
    input_layer: eqx.Module

    def __init__(self, action_head, torso, input_layer=None, *, key=None):
        if input_layer is None:
            input_layer = ObservationInput()
        self.action_head = action_head
        self.torso = torso
        self.input_layer = input_layer

    def __call__(self, observation: Observation) -> distrax.DistributionLike:
        ##print(f"Model ID: {id(self)} | Shape: {observation.agent_view.shape}")
        obs_embedding = self.input_layer(observation)
        obs_embedding = self.torso(obs_embedding)
        return self.action_head(obs_embedding)


class FeedForwardStochasticActor(eqx.Module):
    action_head: eqx.Module
    torso: eqx.Module
    input_layer: eqx.Module

    def __init__(self, action_head, torso, input_layer=None):
        if input_layer is None:
            input_layer = ObservationInput()
        self.action_head = action_head
        self.torso = torso
        self.input_layer = input_layer

    def __call__(self, observation: Observation, key):
        key1, key2 = jax.random.split(key)
        obs_embedding = self.input_layer(observation)
        obs_embedding = self.torso(obs_embedding, key=key1)
        return self.action_head(obs_embedding, key=key2)


class FeedForwardCritic(eqx.Module):
    """Simple Feedforward Critic Network."""

    critic_head: eqx.Module
    torso: eqx.Module
    input_layer: eqx.Module

    def __init__(self, critic_head, torso, input_layer=None, *, key=None):
        if input_layer is None:
            input_layer = ObservationInput()
        self.critic_head = critic_head
        self.torso = torso
        self.input_layer = input_layer

    def __call__(self, observation: Observation) -> jnp.ndarray:
        obs_embedding = self.input_layer(observation)
        obs_embedding = self.torso(obs_embedding)
        critic_output = self.critic_head(obs_embedding)
        return critic_output


class CompositeNetwork(eqx.Module):
    """Composite Network. Takes in a sequence of layers and applies them sequentially."""

    layers: Sequence[eqx.Module]

    def __init__(self, layers):
        self.layers = layers

    def __call__(
        self, *network_input: Union[chex.Array, Tuple[chex.Array, ...]]
    ) -> Union[distrax.DistributionLike, chex.Array]:
        x = self.layers[0](*network_input)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class MultiNetwork(eqx.Module):
    """Multi Network.

    Takes in a sequence of networks, applies them separately and concatenates the outputs."""

    networks: Sequence[eqx.Module]

    def __init__(self, networks):
        self.networks = networks

    def __call__(
        self, *network_input: Union[chex.Array, Tuple[chex.Array, ...]]
    ) -> Union[distrax.DistributionLike, chex.Array]:
        outputs = []
        for network in self.networks:
            outputs.append(network(*network_input))
        concatenated = jnp.stack(outputs, axis=0)  # stack along networks
        chex.assert_rank(concatenated, 1)
        return concatenated


'''
class ScannedRNN(nn.Module):
    hidden_state_dim: int
    cell_type: str

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, rnn_state: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        ins, resets = x
        hidden_state_reset_fn = lambda reset_state, current_state: jnp.where(
            resets[:, np.newaxis],
            reset_state,
            current_state,
        )
        rnn_state = jax.tree.map(
            hidden_state_reset_fn,
            self.initialize_carry(ins.shape[0]),
            rnn_state,
        )
        new_rnn_state, y = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)(
            rnn_state, ins
        )
        return new_rnn_state, y

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, self.hidden_state_dim))



class RecurrentActor(nn.Module):
    """Recurrent Actor Network."""

    action_head: nn.Module
    post_torso: nn.Module
    hidden_state_dim: int
    cell_type: str
    pre_torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, distrax.DistributionLike]:

        observation, done = observation_done

        observation = self.input_layer(observation)
        policy_embedding = self.pre_torso(observation)
        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN(self.hidden_state_dim, self.cell_type)(
            policy_hidden_state, policy_rnn_input
        )
        actor_logits = self.post_torso(policy_embedding)
        pi = self.action_head(actor_logits)

        return policy_hidden_state, pi


class RecurrentCritic(nn.Module):
    """Recurrent Critic Network."""

    critic_head: nn.Module
    post_torso: nn.Module
    hidden_state_dim: int
    cell_type: str
    pre_torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, chex.Array]:

        observation, done = observation_done

        observation = self.input_layer(observation)

        critic_embedding = self.pre_torso(observation)
        critic_rnn_input = (critic_embedding, done)
        critic_hidden_state, critic_embedding = ScannedRNN(self.hidden_state_dim, self.cell_type)(
            critic_hidden_state, critic_rnn_input
        )
        critic_output = self.post_torso(critic_embedding)
        critic_output = self.critic_head(critic_output)

        return critic_hidden_state, critic_output
'''


def chained_torsos(torso_cfgs: List[Dict[str, Any]]) -> eqx.Module:
    """Create a network by chaining multiple torsos together using a list of configs.
    This makes use of hydra to instantiate the modules and the composite network
    to chain them together.

    Args:
        torso_cfgs: List of dictionaries containing the configuration for each torso.
            These configs should use the same format as the individual torso configs."""

    torso_modules = [hydra.utils.instantiate(torso_cfg) for torso_cfg in torso_cfgs]
    return CompositeNetwork(torso_modules)
