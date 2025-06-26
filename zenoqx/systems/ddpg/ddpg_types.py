import optax
from typing_extensions import NamedTuple

from zenoqx.base_types import OnlineAndTarget


class DDPGModels(NamedTuple):
    actor_models: OnlineAndTarget
    q_models: OnlineAndTarget


class DDPGOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState
