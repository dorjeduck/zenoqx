from typing import Dict, Union

import chex
import equinox as eqx
import optax
from flashbax.buffers.trajectory_buffer import BufferState
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from zenoqx.base_types import Action, Done, LogEnvState, OnlineAndTarget, Truncated


class SequenceStep(NamedTuple):
    obs: chex.ArrayTree
    action: Action
    reward: chex.Array
    done: Done
    truncated: Truncated
    log_prob: chex.Array
    info: Dict


class DualParams(NamedTuple):
    log_temperature: chex.Array
    log_alpha_mean: chex.Array
    log_alpha_stddev: chex.Array


class CategoricalDualParams(NamedTuple):
    log_temperature: chex.Array
    log_alpha: chex.Array


class MPOModels(NamedTuple):
    actor_models: OnlineAndTarget
    q_models: OnlineAndTarget
    dual_params: Union[DualParams, CategoricalDualParams]


class MPOOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState
    dual_opt_state: optax.OptState


class MPOLearnerState(NamedTuple):
    models: MPOModels
    opt_states: MPOOptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class VMPOModels(NamedTuple):
    actor_models: OnlineAndTarget
    critic_model: eqx.Module
    dual_params: Union[DualParams, CategoricalDualParams]


class VMPOOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    dual_opt_state: optax.OptState


class VMPOLearnerState(NamedTuple):
    models: VMPOModels
    opt_states: VMPOOptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    learner_step_count: int
