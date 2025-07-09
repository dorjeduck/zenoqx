import copy
import time
from typing import Any, Callable, Dict, Tuple

import chex
import equinox as eqx
import flashbax as fbx
import hydra
import jax
import jax.numpy as jnp
import optax
import rlax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from zenoqx.base_types import (
    AnakinExperimentOutput,
    LearnerFn,
    LogEnvState,
    OnlineAndTarget,
)
from zenoqx.evaluator import evaluator_setup, get_distribution_act_fn
from zenoqx.networks.base import CompositeNetwork
from zenoqx.networks.base import FeedForwardActor as Actor
from zenoqx.systems.mpo.discrete_loss import (
    categorical_mpo_loss,
    clip_categorical_mpo_params,
)
from zenoqx.systems.mpo.mpo_types import (
    CategoricalDualParams,
    MPOLearnerState,
    MPOOptStates,
    MPOModels,
    SequenceStep,
)
from zenoqx.utils import make_env as environments
from zenoqx.utils.checkpointing import Checkpointer
from zenoqx.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from zenoqx.utils.logger import LogEvent, ZenoqxLogger
from zenoqx.utils.multistep import (
    batch_n_step_bootstrapped_returns,
    batch_retrace_continuous,
    batch_truncated_generalized_advantage_estimation,
)
from zenoqx.utils.total_timestep_checker import check_total_timesteps
from zenoqx.utils.training import make_learning_rate
from zenoqx.wrappers.episode_metrics import get_final_step_metrics


def get_warmup_fn(
    env: Environment,
    models: MPOModels,
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:
    def warmup(
        env_states: LogEnvState,
        timesteps: TimeStep,
        buffer_states: BufferState,
        keys: chex.PRNGKey,
    ) -> Tuple[LogEnvState, TimeStep, BufferState, chex.PRNGKey]:
        def _env_step(
            carry: Tuple[LogEnvState, TimeStep, chex.PRNGKey], _: Any
        ) -> Tuple[Tuple[LogEnvState, TimeStep, chex.PRNGKey], SequenceStep]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = models.actor_models.online(last_timestep.observation)
            action = actor_policy.sample(key=policy_key)
            # Ensure action is int32 to match buffer dtype
            action = action.astype(jnp.int32)
            log_prob = actor_policy.log_prob(action)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            sequence_step = SequenceStep(
                last_timestep.observation, action, timestep.reward, done, truncated, log_prob, info
            )

            return (env_state, timestep, key), sequence_step

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (env_states, timesteps, keys), traj_batch = jax.lax.scan(
            _env_step, (env_states, timesteps, keys), None, config.system.warmup_steps
        )

        # Add the trajectory to the buffer.
        # Swap the batch and time axes.
        traj_batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_states = buffer_add_fn(buffer_states, traj_batch)

        return env_states, timesteps, keys, buffer_states

    batched_warmup_step: Callable = jax.vmap(
        warmup, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0), axis_name="batch"
    )

    return batched_warmup_step


def get_learner_fn(
    env: Environment,
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn, optax.TransformUpdateFn],
    buffer_fns: Tuple[Callable, Callable],
    config: DictConfig,
) -> LearnerFn[MPOLearnerState]:
    """Get the learner function."""

    # Get update functions for actor and critic models.
    actor_update_fn, q_update_fn, dual_update_fn = update_fns
    buffer_add_fn, buffer_sample_fn = buffer_fns

    def _update_step(learner_state: MPOLearnerState, _: Any) -> Tuple[MPOLearnerState, Tuple]:
        def _env_step(
            learner_state: MPOLearnerState, _: Any
        ) -> Tuple[MPOLearnerState, SequenceStep]:
            """Step the environment."""
            models, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = models.actor_models.online(last_timestep.observation)
            action = actor_policy.sample(key=policy_key)
            # Ensure action is int32 to match buffer dtype
            action = action.astype(jnp.int32)
            log_prob = actor_policy.log_prob(action)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            sequence_step = SequenceStep(
                last_timestep.observation, action, timestep.reward, done, truncated, log_prob, info
            )

            learner_state = MPOLearnerState(
                models, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, sequence_step

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        models, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        # Add the trajectory to the buffer.
        # Swap the batch and time axes.
        traj_batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the model for a single epoch."""

            def _actor_loss_fn(
                online_actor_model: eqx.Module,
                dual_params: CategoricalDualParams,
                target_actor_model: eqx.Module,
                target_q_model: eqx.Module,
                sequence: SequenceStep,
            ) -> chex.Array:
                # Reshape the observations to [B*T, ...].
                reshaped_obs = jax.tree.map(lambda x: merge_leading_dims(x, 2), sequence.obs)
                batch_length = sequence.action.shape[0] * sequence.action.shape[1]  # B*T

                online_actor_policy = online_actor_model(reshaped_obs)
                target_actor_policy = target_actor_model(reshaped_obs)
                # In discrete MPO, we evaluate all actions instead of sampling.
                a_improvement = jnp.arange(config.system.action_dim).astype(jnp.float32)
                a_improvement = jnp.tile(
                    a_improvement[..., jnp.newaxis], [1, batch_length]
                )  # [D, B*T]
                a_improvement = jax.nn.one_hot(a_improvement, config.system.action_dim)

                target_q_values = jax.vmap(target_q_model, in_axes=(None, 0))(
                    reshaped_obs, a_improvement
                )

                # Compute the policy and dual loss.
                loss, loss_info = categorical_mpo_loss(
                    dual_params=dual_params,
                    online_action_distribution=online_actor_policy,
                    target_action_distribution=target_actor_policy,
                    q_values=target_q_values,
                    epsilon=config.system.epsilon,
                    epsilon_policy=config.system.epsilon_policy,
                )

                return jnp.mean(loss), loss_info

            def _q_loss_fn(
                online_q_model: eqx.Module,
                target_q_model: eqx.Module,
                online_actor_model: eqx.Module,
                target_actor_model: eqx.Module,
                sequence: SequenceStep,
                key: chex.PRNGKey,
            ) -> jnp.ndarray:

                online_actor_policy = jax.vmap(online_actor_model)(sequence.obs)  # [B, T, ...]
                target_actor_policy = jax.vmap(target_actor_model)(sequence.obs)  # [B, T, ...]
                a_t = jax.nn.one_hot(sequence.action, config.system.action_dim)  # [B, T, ...]
                online_q_t = jax.vmap(online_q_model)(sequence.obs, a_t)  # [B, T]

                # Cast and clip rewards.
                discount = 1.0 - sequence.done.astype(jnp.float32)
                d_t = (discount * config.system.gamma).astype(jnp.float32)
                r_t = jnp.clip(
                    sequence.reward, -config.system.max_abs_reward, config.system.max_abs_reward
                ).astype(jnp.float32)

                # Policy to use for policy evaluation and bootstrapping.
                if config.system.use_online_policy_to_bootstrap:
                    policy_to_evaluate = online_actor_policy
                else:
                    policy_to_evaluate = target_actor_policy

                # Action(s) to use for policy evaluation; shape [N, B, T].
                if config.system.stochastic_policy_eval:

                    sample_keys = jax.random.split(key, config.system.num_samples)
                    a_evaluation = jax.vmap(
                        lambda k: jax.vmap(lambda p: p.sample(key=k))(policy_to_evaluate)
                    )(
                        sample_keys
                    )  # [N, B, T, ...]
                else:
                    a_evaluation = policy_to_evaluate.mode()[jnp.newaxis, ...]  # [N=1, B, T, ...]

                # Add a stopgrad in case we use the online policy for evaluation.
                a_evaluation = jax.lax.stop_gradient(a_evaluation)
                a_evaluation = jax.nn.one_hot(a_evaluation, config.system.action_dim)

                # Compute the Q-values for the next state-action pairs; [N, B, T].
                q_values = jax.vmap(jax.vmap(target_q_model), in_axes=(None, 0))(
                    sequence.obs, a_evaluation
                )

                # When policy_eval_stochastic == True, this corresponds to expected SARSA.
                # Otherwise, the mean is a no-op.
                v_t = jnp.mean(q_values, axis=0)  # [B, T]

                if config.system.use_retrace:
                    # Compute the log-rhos for the retrace targets.
                    log_rhos = target_actor_policy.log_prob(sequence.action) - sequence.log_prob

                    # Compute target Q-values
                    target_q_t = target_q_model(sequence.obs, a_t)  # [B, T]

                    # Compute retrace targets.
                    # These targets use the rewards and discounts as in normal TD-learning but
                    # they use a mix of bootstrapped values V(s') and Q(s', a'), weighing the
                    # latter based on how likely a' is under the current policy (s' and a' are
                    # samples from replay).
                    # See [Munos et al., 2016](https://arxiv.org/abs/1606.02647) for more.
                    retrace_error = batch_retrace_continuous(
                        online_q_t[:, :-1],
                        target_q_t[:, 1:-1],
                        v_t[:, 1:],
                        r_t[:, :-1],
                        d_t[:, :-1],
                        log_rhos[:, 1:-1],
                        config.system.retrace_lambda,
                    )
                    q_loss = rlax.l2_loss(retrace_error).mean()
                elif config.system.use_n_step_bootstrap:
                    n_step_value_target = batch_n_step_bootstrapped_returns(
                        r_t[:, :-1],
                        d_t[:, :-1],
                        v_t[:, 1:],
                        config.system.n_step_for_sequence_bootstrap,
                    )
                    td_error = online_q_t[:, :-1] - n_step_value_target
                    q_loss = rlax.l2_loss(td_error).mean()
                else:
                    _, gae_value_target = batch_truncated_generalized_advantage_estimation(
                        r_t[:, :-1],
                        d_t[:, :-1],
                        config.system.gae_lambda,
                        v_t,
                        time_major=False,
                        truncation_t=sequence.truncated[:, :-1],
                    )
                    td_error = online_q_t[:, :-1] - gae_value_target
                    q_loss = rlax.l2_loss(td_error).mean()

                loss_info = {
                    "q_loss": q_loss,
                }

                return q_loss, loss_info

            models, opt_states, buffer_state, key = update_state

            key, sample_key, q_key = jax.random.split(key, num=3)

            # SAMPLE SEQUENCES
            sequence_sample = buffer_sample_fn(buffer_state, sample_key)
            sequence: SequenceStep = sequence_sample.experience

            # CALCULATE ACTOR AND DUAL LOSS
            actor_dual_grad_fn = jax.grad(_actor_loss_fn, argnums=(0, 1), has_aux=True)
            actor_dual_grads, actor_dual_loss_info = actor_dual_grad_fn(
                models.actor_models.online,
                models.dual_params,
                models.actor_models.target,
                models.q_models.target,
                sequence,
            )

            # CALCULATE Q LOSS
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                models.q_models.online,
                models.q_models.target,
                models.actor_models.online,
                models.actor_models.target,
                sequence,
                q_key,
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.
            actor_dual_grads, actor_dual_loss_info = jax.lax.pmean(
                (actor_dual_grads, actor_dual_loss_info), axis_name="batch"
            )
            # pmean over devices.
            actor_dual_grads, actor_dual_loss_info = jax.lax.pmean(
                (actor_dual_grads, actor_dual_loss_info), axis_name="device"
            )

            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

            actor_grads, dual_grads = actor_dual_grads

            # UPDATE ACTOR MODEL AND OPTIMISER STATE
            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state
            )
            actor_new_online_model = optax.apply_updates(models.actor_models.online, actor_updates)

            # UPDATE DUAL PARAMS AND OPTIMISER STATE
            dual_updates, dual_new_opt_state = dual_update_fn(dual_grads, opt_states.dual_opt_state)
            dual_new_params = optax.apply_updates(models.dual_params, dual_updates)
            dual_new_params = clip_categorical_mpo_params(dual_new_params)

            # UPDATE Q MODELS AND OPTIMISER STATE
            q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states.q_opt_state)
            q_new_online_model = optax.apply_updates(models.q_models.online, q_updates)
            # Target model polyak update.
            (new_target_actor_model, new_target_q_model) = optax.incremental_update(
                (actor_new_online_model, q_new_online_model),
                (models.actor_models.target, models.q_models.target),
                config.system.tau,
            )

            actor_new_models = OnlineAndTarget(actor_new_online_model, new_target_actor_model)
            q_new_models = OnlineAndTarget(q_new_online_model, new_target_q_model)

            # PACK NEW MODELS AND OPTIMISER STATE
            new_models = MPOModels(actor_new_models, q_new_models, dual_new_params)
            new_opt_state = MPOOptStates(actor_new_opt_state, q_new_opt_state, dual_new_opt_state)

            # PACK LOSS INFO
            loss_info = {
                **actor_dual_loss_info,
                **q_loss_info,
            }
            return (new_models, new_opt_state, buffer_state, key), loss_info

        update_state = (models, opt_states, buffer_state, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        models, opt_states, buffer_state, key = update_state
        learner_state = MPOLearnerState(
            models, opt_states, buffer_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: MPOLearnerState) -> AnakinExperimentOutput[MPOLearnerState]:
        """Learner function.

        This function represents the learner, it updates the model
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, key: chex.PRNGKey, config: DictConfig
) -> Tuple[LearnerFn[MPOLearnerState], Actor, MPOLearnerState]:
    """Initialise learner_fn, model, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of observations.
    observation_dim = env.observation_spec().agent_view.shape[0]
    config.system.observation_dim = observation_dim

    # Get number of actions or action dimension from the environment.
    action_dim = int(env.action_spec().num_values)
    config.system.action_dim = action_dim

    # PRNG keys.
    key, *keys = jax.random.split(key, 5)

    # Define and init actor_model, q_model and optimiser.
    actor_torso = hydra.utils.instantiate(
        config.network.actor_network.pre_torso, input_dim=observation_dim, key=keys[0]
    )
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        input_dim=actor_torso.output_dim,
        action_dim=action_dim,
        key=keys[1],
    )
    actor_model = Actor(torso=actor_torso, action_head=actor_action_head)

    q_model_input = hydra.utils.instantiate(config.network.q_network.input_layer)
    q_model_torso = hydra.utils.instantiate(
        config.network.q_network.pre_torso, input_dim=observation_dim + action_dim, key=keys[2]
    )
    q_model_head = hydra.utils.instantiate(
        config.network.q_network.critic_head, input_dim=q_model_torso.output_dim, key=keys[3]
    )
    q_model = CompositeNetwork([q_model_input, q_model_torso, q_model_head])

    actor_lr = make_learning_rate(config.system.actor_lr, config, config.system.epochs)
    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(q_lr, eps=1e-5),
    )

    # Initialise optimiser states.
    actor_opt_state = actor_optim.init(actor_model)
    q_opt_state = q_optim.init(q_model)

    log_temperature = jnp.full([1], config.system.init_log_temperature, dtype=jnp.float32)
    log_alpha = jnp.full([1], config.system.init_log_alpha, dtype=jnp.float32)

    dual_params = CategoricalDualParams(
        log_temperature=log_temperature,
        log_alpha=log_alpha,
    )

    dual_lr = make_learning_rate(config.system.dual_lr, config, config.system.epochs)
    dual_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(dual_lr, eps=1e-5),
    )
    dual_opt_state = dual_optim.init(dual_params)

    models = MPOModels(
        OnlineAndTarget(actor_model, actor_model),
        OnlineAndTarget(q_model, q_model),
        dual_params,
    )
    opt_states = MPOOptStates(actor_opt_state, q_opt_state, dual_opt_state)

    # Pack update functions.
    update_fns = (actor_optim.update, q_optim.update, dual_optim.update)

    # Initialise observation
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree.map(lambda x: x[None, ...], init_x)

    # Create replay buffer
    dummy_sequence_step = SequenceStep(
        obs=jax.tree.map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        truncated=jnp.zeros((), dtype=bool),
        log_prob=jnp.zeros((), dtype=float),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
    )

    assert config.system.total_buffer_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total buffer size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    assert config.system.total_batch_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total batch size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    config.system.buffer_size = config.system.total_buffer_size // (
        n_devices * config.arch.update_batch_size
    )
    config.system.batch_size = config.system.total_batch_size // (
        n_devices * config.arch.update_batch_size
    )
    buffer_fn = fbx.make_trajectory_buffer(
        max_size=config.system.buffer_size,
        min_length_time_axis=config.system.sample_sequence_length,
        sample_batch_size=config.system.batch_size,
        sample_sequence_length=config.system.sample_sequence_length,
        period=config.system.period,
        add_batch_size=config.arch.num_envs,
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample)
    buffer_states = buffer_fn.init(dummy_sequence_step)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, update_fns, buffer_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    warmup = get_warmup_fn(env, models, buffer_fn.add, config)
    warmup = jax.pmap(warmup, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree.map(reshape_states, env_states)
    timesteps = jax.tree.map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        models, _ = loaded_checkpoint.restore_models(template_models=models)

    # Define params to be replicated across devices and batches.
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))

    replicate_learner = (models, opt_states, buffer_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = jax.device_put_replicated(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    models, opt_states, buffer_states = replicate_learner
    # Warmup the buffer.
    env_states, timesteps, keys, buffer_states = warmup(
        env_states, timesteps, buffer_states, warmup_keys
    )
    init_learner_state = MPOLearnerState(
        models, opt_states, buffer_states, step_keys, env_states, timesteps
    )

    return learn, actor_model, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config)

    # PRNG keys.
    key, key_e, key_l = jax.random.split(jax.random.key(config.arch.seed), num=3)

    # Setup learner.
    learn, actor_model, learner_state = learner_setup(env, key_l, config)

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key=key_e,
        eval_act_fn=get_distribution_act_fn(config),
        model=learner_state.models.actor_models.online,
        config=config,
    )

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = ZenoqxLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = jnp.float32(-1e6)
    best_model = unreplicate_batch_dim(learner_state.models.actor_models.online)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval * (config.system.epochs)
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.models.actor_models.online
        )  # Select only actor model
        key, *eval_keys = jax.random.split(key, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_model = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key, *eval_keys = jax.random.split(key, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)

        evaluator_output = absolute_metric_evaluator(best_model, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run. If the absolute metric is not
    # calculated, this will be the final evaluation run.
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_ff_mpo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}MPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
