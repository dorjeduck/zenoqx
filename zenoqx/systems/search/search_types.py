from typing import Callable, Dict, Tuple, Union

import chex
import equinox as eqx
import mctx
from distrax import DistributionLike
from jumanji.types import TimeStep
from optax import OptState
from typing_extensions import NamedTuple

from zenoqx.base_types import Action, ActorCriticModels, Done, Observation, Value

SearchApply = Callable[[eqx.Module, chex.PRNGKey, mctx.RootFnOutput], mctx.PolicyOutput]
RootFnApply = Callable[
    [eqx.Module, Observation, chex.ArrayTree, chex.PRNGKey], mctx.RootFnOutput
]
EnvironmentStep = Callable[[chex.ArrayTree, Action], Tuple[chex.ArrayTree, TimeStep]]

RepresentationApply = Callable[[eqx.Module, Observation], chex.Array]
DynamicsApply = Callable[[eqx.Module, chex.Array, chex.Array], Tuple[chex.Array, DistributionLike]]


class ExItTransition(NamedTuple):
    done: Done
    action: Action
    reward: chex.Array
    search_value: Value
    search_policy: chex.Array
    obs: chex.Array
    info: Dict


class SampledExItTransition(NamedTuple):
    done: chex.Array
    action: Action
    sampled_actions: chex.Array
    reward: chex.Array
    search_value: Value
    search_policy: chex.Array
    obs: chex.Array
    info: Dict


class MZModels(NamedTuple):
    prediction_models: ActorCriticModels
    worldmodel_model: eqx.Module


class ZLearnerState(NamedTuple):
    models: Union[MZModels, ActorCriticModels]
    opt_states: OptState
    buffer_state: chex.ArrayTree
    key: chex.PRNGKey
    env_state: TimeStep
    timestep: TimeStep
