import chex
import equinox as eqx
import optax
from typing_extensions import NamedTuple

from zenoqx.base_types import OnlineAndTarget


class SACModels(NamedTuple):
    actor_model: eqx.Module
    q_model: OnlineAndTarget
    log_alpha: chex.Array


class SACOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState
    alpha_opt_state: optax.OptState
