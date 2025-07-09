import chex
import equinox as eqx
import jax

from flashbax.buffers.trajectory_buffer import BufferState
from jumanji.types import TimeStep
from optax import OptState
from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, Tuple, TypeVar
from typing_extensions import NamedTuple, TypeAlias

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    import flax
    from flax.struct import dataclass

Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
Truncated: TypeAlias = chex.Array
First: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array
# Can't know the exact type of State.
State: TypeAlias = Any
Models: TypeAlias = Any
OptStates: TypeAlias = Any
HiddenStates: TypeAlias = Any


class Observation(NamedTuple):
    """The observation that the agent sees.
    agent_view: the agent's view of the environment.
    action_mask: boolean array specifying which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_view: chex.Array  # (num_obs_features,)
    action_mask: chex.Array  # (num_actions,)
    step_count: Optional[chex.Array] = None  # (,)


class ObservationGlobalState(NamedTuple):
    """The observation seen by agents in centralised systems.
    Extends `Observation` by adding a `global_state` attribute for centralised training.
    global_state: The global state of the environment, often a concatenation of agents' views.
    """

    agent_view: chex.Array
    action_mask: chex.Array
    global_state: chex.Array
    step_count: chex.Array


@dataclass
class LogEnvState:
    """State of the `LogWrapper`."""

    env_state: State
    episode_returns: chex.Numeric
    episode_lengths: chex.Numeric
    # Information about the episode return and length for logging purposes.
    episode_return_info: chex.Numeric
    episode_length_info: chex.Numeric


class EvalState(NamedTuple):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    step_count: chex.Array
    episode_return: chex.Array


class RNNEvalState(NamedTuple):
    """State of the evaluator for recurrent architectures."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    dones: Done
    hstate: HiddenState
    step_count: chex.Array
    episode_return: chex.Array


class ActorCriticModels(NamedTuple):
    """Models of an actor critic network."""

    actor_model: eqx.Module
    critic_model: eqx.Module


class ActorCriticOptStates(NamedTuple):
    """OptStates of actor critic learner."""

    actor_opt_state: OptState
    critic_opt_state: OptState


class ActorCriticHiddenStates(NamedTuple):
    """Hidden states for an actor critic learner."""

    policy_hidden_state: HiddenState
    critic_hidden_state: HiddenState


class CoreLearnerState(NamedTuple):
    """Base state of the learner. Can be used for both on-policy and off-policy learners.
    Mainly used for sebulba systems since we dont store env state."""

    models: Models
    opt_states: OptStates
    key: chex.PRNGKey
    timestep: TimeStep


class OnPolicyLearnerState(NamedTuple):
    """State of the learner. Used for on-policy learners."""

    models: Models
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class RNNLearnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures."""

    models: Models
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    done: Done
    truncated: Truncated
    hstates: HiddenStates


class OffPolicyLearnerState(NamedTuple):
    models: eqx.Module
    opt_states: OptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class RNNOffPolicyLearnerState(NamedTuple):
    params: Models
    opt_states: OptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    dones: Done
    truncated: Truncated
    hstates: HiddenStates


class OnlineAndTarget(NamedTuple):
    online: eqx.Module
    target: eqx.Module


ZenoqxState = TypeVar(
    "ZenoqxState",
)
ZenoqxTransition = TypeVar(
    "ZenoqxTransition",
)


class SebulbaExperimentOutput(NamedTuple, Generic[ZenoqxState]):
    """Experiment output."""

    learner_state: ZenoqxState
    train_metrics: Dict[str, chex.Array]


class AnakinExperimentOutput(NamedTuple, Generic[ZenoqxState]):
    """Experiment output."""

    learner_state: ZenoqxState
    episode_metrics: Dict[str, chex.Array]
    train_metrics: Dict[str, chex.Array]


class EvaluationOutput(NamedTuple, Generic[ZenoqxState]):
    """Evaluation output."""

    learner_state: ZenoqxState
    episode_metrics: Dict[str, chex.Array]


RNNObservation: TypeAlias = Tuple[Observation, Done]
LearnerFn = Callable[[ZenoqxState], AnakinExperimentOutput[ZenoqxState]]
SebulbaLearnerFn = Callable[[ZenoqxState, ZenoqxTransition], SebulbaExperimentOutput[ZenoqxState]]
EvalFn = Callable[[eqx.Module, chex.PRNGKey], EvaluationOutput[ZenoqxState]]
SebulbaEvalFn = Callable[[eqx.Module, chex.PRNGKey], Dict[str, chex.Array]]


ActFn = Callable[[Observation, chex.PRNGKey], chex.Array]
CriticApply = Callable[[Observation], Value]
ContinuousQApply = Callable[[Observation, Action], Value]

RecActFn = Callable[
    [HiddenState, RNNObservation, chex.PRNGKey], Tuple[HiddenState, chex.Array]
]
RecCriticApply = Callable[[HiddenState, RNNObservation], Tuple[HiddenState, Value]]
