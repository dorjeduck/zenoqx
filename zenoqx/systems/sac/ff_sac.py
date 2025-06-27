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
    OffPolicyLearnerState,
    OnlineAndTarget,
)
from zenoqx.evaluator import evaluator_setup, get_distribution_act_fn
from zenoqx.networks.base import CompositeNetwork
from zenoqx.networks.base import FeedForwardActor as Actor
from zenoqx.networks.base import MultiNetwork
from zenoqx.systems.q_learning.dqn_types import Transition
from zenoqx.systems.sac.sac_types import SACOptStates, SACModels
from zenoqx.utils import make_env as environments
from zenoqx.utils.checkpointing import Checkpointer
from zenoqx.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from zenoqx.utils.logger import LogEvent, ZenoqxLogger
from zenoqx.utils.total_timestep_checker import check_total_timesteps
from zenoqx.utils.training import make_learning_rate
from zenoqx.wrappers.episode_metrics import get_final_step_metrics


def get_warmup_fn(
    env: Environment,
    models: SACModels,
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:
    def warmup(
        env_states: LogEnvState,
        timesteps: TimeStep,
        buffer_states: BufferState,
        keys: chex.PRNGKey,
    ) -> Tuple[LogEnvState, TimeStep, BufferState, jax.random.PRNGKey]:
        def _env_step(
            carry: Tuple[LogEnvState, TimeStep, jax.random.PRNGKey], _: Any
        ) -> Tuple[Tuple[LogEnvState, TimeStep, jax.random.PRNGKey], Transition]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = jax.vmap(models.actor_model)(last_timestep.observation)  ## [1024,8]
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step)(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]

            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            return (env_state, timestep, key), transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (env_states, timesteps, keys), traj_batch = jax.lax.scan(
            _env_step, (env_states, timesteps, keys), None, config.system.warmup_steps
        )

        # Add the trajectory to the buffer.
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
) -> LearnerFn[OffPolicyLearnerState]:
    """Get the learner function."""

    # Get update functions for actor and critic models.
    actor_update_fn, q_update_fn, alpha_update_fn = update_fns
    buffer_add_fn, buffer_sample_fn = buffer_fns

    def _update_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: OffPolicyLearnerState, _: Any
        ) -> Tuple[OffPolicyLearnerState, Transition]:
            """Step the environment."""
            models, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = jax.vmap(models.actor_model)(last_timestep.observation)

            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step)(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]

            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            learner_state = OffPolicyLearnerState(
                models, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        models, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        # Add the trajectory to the buffer.
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the model for a single epoch."""

            def _alpha_loss_fn(
                log_alpha: chex.Array,
                actor_model: eqx.Module,
                transitions: Transition,
                key: chex.PRNGKey,
            ) -> jnp.ndarray:
                """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
                actor_policy = jax.vmap(actor_model)(transitions.obs)
                action = actor_policy.sample(seed=key)
                log_prob = actor_policy.log_prob(action)
                alpha = jnp.exp(log_alpha)
                alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - config.system.target_entropy)

                loss_info = {
                    "alpha_loss": jnp.mean(alpha_loss),
                    "alpha": jnp.mean(alpha),
                }
                return jnp.mean(alpha_loss), loss_info

            def _q_loss_fn(
                q_model: eqx.Module,
                actor_model: eqx.Module,
                target_q_model: eqx.Module,
                alpha: jnp.ndarray,
                transitions: Transition,
                key: chex.PRNGKey,
            ) -> jnp.ndarray:
                q_old_action = jax.vmap(q_model)(transitions.obs, transitions.action)
                next_actor_policy = jax.vmap(actor_model)(transitions.next_obs)
                next_action = next_actor_policy.sample(seed=key)
                next_log_prob = next_actor_policy.log_prob(next_action)
                next_q = jax.vmap(target_q_model)(transitions.next_obs, next_action)
                next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
                target_q = jax.lax.stop_gradient(
                    transitions.reward + (1.0 - transitions.done) * config.system.gamma * next_v
                )
                q_error = q_old_action - jnp.expand_dims(target_q, -1)
                q_loss = 0.5 * jnp.mean(jnp.square(q_error))

                loss_info = {
                    "q_loss": jnp.mean(q_loss),
                    "q_error": jnp.mean(jnp.abs(q_error)),
                    "q1_pred": jnp.mean(next_q[..., 0]),
                    "q2_pred": jnp.mean(next_q[..., 1]),
                }
                return q_loss, loss_info

            def _actor_loss_fn(
                actor_model: eqx.Module,
                q_model: eqx.Module,
                alpha: chex.Array,
                transitions: Transition,
                key: chex.PRNGKey,
            ) -> chex.Array:
                actor_policy = jax.vmap(actor_model)(transitions.obs)
                action = actor_policy.sample(seed=key)
                log_prob = actor_policy.log_prob(action)

                q_action = jax.vmap(q_model)(transitions.obs, action)
                min_q = jnp.min(q_action, axis=-1)
                actor_loss = alpha * log_prob - min_q

                loss_info = {
                    "actor_loss": jnp.mean(actor_loss),
                    "entropy": jnp.mean(-log_prob),
                }
                return jnp.mean(actor_loss), loss_info

            models, opt_states, buffer_state, key = update_state

            key, sample_key, actor_key, q_key, alpha_key = jax.random.split(key, num=5)

            # SAMPLE TRANSITIONS
            transition_sample = buffer_sample_fn(buffer_state, sample_key)
            transitions: Transition = transition_sample.experience
            alpha = jnp.exp(models.log_alpha)

            # CALCULATE ACTOR LOSS
            actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
            actor_grads, actor_loss_info = actor_grad_fn(
                models.actor_model, models.q_model.online, alpha, transitions, actor_key
            )

            # CALCULATE Q LOSS
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                models.q_model.online,
                models.actor_model,
                models.q_model.target,
                alpha,
                transitions,
                q_key,
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.
            actor_grads, actor_loss_info = jax.lax.pmean(
                (actor_grads, actor_loss_info), axis_name="batch"
            )
            # pmean over devices.
            actor_grads, actor_loss_info = jax.lax.pmean(
                (actor_grads, actor_loss_info), axis_name="device"
            )

            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

            if config.system.autotune:
                alpha_grad_fn = jax.grad(_alpha_loss_fn, has_aux=True)
                alpha_grads, alpha_loss_info = alpha_grad_fn(
                    models.log_alpha, models.actor_model, transitions, alpha_key
                )
                alpha_grads, alpha_loss_info = jax.lax.pmean(
                    (alpha_grads, alpha_loss_info), axis_name="batch"
                )
                alpha_grads, alpha_loss_info = jax.lax.pmean(
                    (alpha_grads, alpha_loss_info), axis_name="device"
                )
                log_alpha_updates, alpha_new_opt_state = alpha_update_fn(
                    alpha_grads, opt_states.alpha_opt_state
                )
                log_alpha_new_model = optax.apply_updates(models.log_alpha, log_alpha_updates)
            else:
                log_alpha_new_model = models.log_alpha
                alpha_new_opt_state = opt_states.alpha_opt_state
                alpha_loss_info = {"alpha_loss": 0.0, "alpha": alpha}

            # UPDATE ACTOR MODEL AND OPTIMISER STATE
            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state
            )
            actor_new_model = optax.apply_updates(models.actor_model, actor_updates)

            # UPDATE Q MODEL AND OPTIMISER STATE
            q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states.q_opt_state)
            q_new_online_model = optax.apply_updates(models.q_model.online, q_updates)
            # Target model polyak update.
            new_target_q_model = optax.incremental_update(
                q_new_online_model, models.q_model.target, config.system.tau
            )
            q_new_model = OnlineAndTarget(q_new_online_model, new_target_q_model)

            # PACK NEW MODEL AND OPTIMISER STATE
            new_model = SACModels(actor_new_model, q_new_model, log_alpha_new_model)
            new_opt_state = SACOptStates(actor_new_opt_state, q_new_opt_state, alpha_new_opt_state)

            # PACK LOSS INFO
            loss_info = {
                **actor_loss_info,
                **q_loss_info,
                **alpha_loss_info,
            }
            return (new_model, new_opt_state, buffer_state, key), loss_info

        update_state = (models, opt_states, buffer_state, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        models, opt_states, buffer_state, key = update_state
        learner_state = OffPolicyLearnerState(
            models, opt_states, buffer_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OffPolicyLearnerState,
    ) -> AnakinExperimentOutput[OffPolicyLearnerState]:
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
) -> Tuple[LearnerFn[OffPolicyLearnerState], Actor, OffPolicyLearnerState]:
    """Initialise learner_fn, model, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of observations.
    observation_dim = env.observation_spec().agent_view.shape[0]
    config.system.observation_dim = observation_dim

    # Get number of actions or action dimension from the environment.
    action_dim = int(env.action_spec().shape[-1])
    config.system.action_dim = action_dim
    config.system.action_minimum = float(env.action_spec().minimum)
    config.system.action_maximum = float(env.action_spec().maximum)

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
        minimum=config.system.action_minimum,
        maximum=config.system.action_maximum,
        key=keys[1],
    )
    actor_model = Actor(torso=actor_torso, action_head=actor_action_head)

    def create_q_model(cfg: DictConfig) -> CompositeNetwork:
        q_model_input = hydra.utils.instantiate(cfg.network.q_network.input_layer)
        q_model_torso = hydra.utils.instantiate(
            cfg.network.q_network.pre_torso,
            input_dim=observation_dim + action_dim,  # ObservationActionInput
            key=keys[2],
        )
        q_model_head = hydra.utils.instantiate(
            cfg.network.q_network.critic_head, input_dim=q_model_torso.output_dim, key=keys[3]
        )
        return CompositeNetwork([q_model_input, q_model_torso, q_model_head])

    double_q_model = MultiNetwork([create_q_model(config), create_q_model(config)])

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
    actor_opt_state = actor_optim.init(eqx.filter(actor_model, eqx.is_array))

    # Initialise q network optimiser state.
    q_opt_state = q_optim.init(eqx.filter(double_q_model, eqx.is_array))

    # Automatic entropy tuning
    target_entropy = -config.system.target_entropy_scale * action_dim
    if config.system.autotune:
        log_alpha = jnp.zeros_like(target_entropy)
    else:
        log_alpha = jnp.log(config.system.init_alpha)

    config.system.target_entropy = target_entropy

    alpha_lr = make_learning_rate(config.system.alpha_lr, config, config.system.epochs)
    alpha_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(alpha_lr, eps=1e-5),
    )
    alpha_opt_state = alpha_optim.init(log_alpha)

    models = SACModels(actor_model, OnlineAndTarget(double_q_model, double_q_model), log_alpha)
    opt_states = SACOptStates(actor_opt_state, q_opt_state, alpha_opt_state)

    # Pack update functions.
    update_fns = (actor_optim.update, q_optim.update, alpha_optim.update)

    # Create replay buffer
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree.map(lambda x: x[None, ...], init_x)

    dummy_transition = Transition(
        obs=jax.tree.map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((action_dim), dtype=float),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree.map(lambda x: x.squeeze(0), init_x),
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
    buffer_fn = fbx.make_item_buffer(
        max_length=config.system.buffer_size,
        min_length=config.system.batch_size,
        sample_batch_size=config.system.batch_size,
        add_batches=True,
        add_sequences=True,
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample)
    buffer_states = buffer_fn.init(dummy_transition)

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
    init_learner_state = OffPolicyLearnerState(
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
    key, key_e, key_l = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)

    # Setup learner.
    learn, actor_model, learner_state = learner_setup(env, key_l, config)

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_model, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_distribution_act_fn(config, actor_model),
        model=learner_state.models.actor_model,
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
    best_model = unreplicate_batch_dim(learner_state.models.actor_model)
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
        trained_model = unreplicate_batch_dim(
            learner_output.learner_state.models.actor_model
        )  # Select only actor model
        key, *eval_keys = jax.random.split(key, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_model, eval_keys)
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
            best_model = copy.deepcopy(trained_model)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key, *eval_keys = jax.random.split(key, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

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
    config_name="default_ff_sac.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}SAC experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
