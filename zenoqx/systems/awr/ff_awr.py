import copy
import time
from typing import Any, Callable, Dict, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
import rlax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from zenoqx.base_types import (
    ActorCriticOptStates,
    ActorCriticModels,
    AnakinExperimentOutput,
    LearnerFn,
    LogEnvState,
)
from zenoqx.evaluator import evaluator_setup, get_distribution_act_fn
from zenoqx.networks.base import FeedForwardActor as Actor
from zenoqx.networks.base import FeedForwardCritic as Critic
from zenoqx.systems.awr.awr_types import AWRLearnerState, SequenceStep
from zenoqx.utils import make_env as environments
from zenoqx.utils.checkpointing import Checkpointer
from zenoqx.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from zenoqx.utils.logger import LogEvent, ZenoqxLogger
from zenoqx.utils.multistep import batch_truncated_generalized_advantage_estimation
from zenoqx.utils.total_timestep_checker import check_total_timesteps
from zenoqx.utils.training import make_learning_rate
from zenoqx.wrappers.episode_metrics import get_final_step_metrics


def get_warmup_fn(
    env: Environment,
    models: ActorCriticModels,
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
        ) -> Tuple[Tuple[LogEnvState, TimeStep, jax.random.PRNGKey], SequenceStep]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = jax.vmap(models.actor_model)(last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            sequence_step = SequenceStep(
                last_timestep.observation, action, timestep.reward, done, truncated, info
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
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    buffer_fns: Tuple[Callable, Callable],
    config: DictConfig,
) -> LearnerFn[AWRLearnerState]:
    """Get the learner function."""

    # Get update functions for actor and critic models.
    actor_update_fn, critic_update_fn = update_fns
    buffer_add_fn, buffer_sample_fn = buffer_fns

    def _update_step(learner_state: AWRLearnerState, _: Any) -> Tuple[AWRLearnerState, Tuple]:
        def _env_step(
            learner_state: AWRLearnerState, _: Any
        ) -> Tuple[AWRLearnerState, SequenceStep]:
            """Step the environment."""
            models, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = jax.vmap(models.actor_model)(last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            sequence_step = SequenceStep(
                last_timestep.observation, action, timestep.reward, done, truncated, info
            )

            learner_state = AWRLearnerState(
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

        def _update_critic_step(update_state: Tuple, _: Any) -> Tuple:
            def _critic_loss_fn(
                critic_model: FrozenDict,
                observations: chex.Array,
                target_vals: jnp.ndarray,
            ) -> jnp.ndarray:

                pred_v = jax.vmap(jax.vmap(critic_model))(observations)[:, :-1]
                critic_loss = rlax.l2_loss(pred_v, target_vals).mean()

                loss_info = {
                    "critic_loss": critic_loss,
                }

                return critic_loss, loss_info

            models, opt_states, buffer_state, key, static_critic_model = update_state

            key, sample_key = jax.random.split(key)

            # SAMPLE SEQUENCES
            sequence_sample = buffer_sample_fn(buffer_state, sample_key)
            sequence: SequenceStep = sequence_sample.experience

            # CALCULATE TARGETS USING LAST ITERATION CRITIC
            v_t = jax.vmap(jax.vmap(static_critic_model))(sequence.obs)
            r_t = sequence.reward[:, :-1]
            d_t = (1 - sequence.done.astype(jnp.float32)[:, :-1]) * config.system.gamma
            _, target_vals = batch_truncated_generalized_advantage_estimation(
                r_t,
                d_t,
                config.system.gae_lambda,
                v_t,
                time_major=False,
                truncation_t=sequence.truncated[:, :-1],
            )

            # CALCULATE CRITIC LOSS
            critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
            critic_grads, critic_loss_info = critic_grad_fn(
                models.critic_model, sequence.obs, target_vals
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.

            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="batch"
            )
            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="device"
            )

            # UPDATE CRITIC MODEL AND OPTIMISER STATE
            critic_updates, critic_new_opt_state = critic_update_fn(
                critic_grads, opt_states.critic_opt_state
            )
            critic_new_model = optax.apply_updates(models.critic_model, critic_updates)

            # PACK NEW MODELS AND OPTIMISER STATE
            new_models = ActorCriticModels(models.actor_model, critic_new_model)
            new_opt_state = ActorCriticOptStates(opt_states.actor_opt_state, critic_new_opt_state)

            return (
                new_models,
                new_opt_state,
                buffer_state,
                key,
                static_critic_model,
            ), critic_loss_info

        def _update_actor_step(update_state: Tuple, _: Any) -> Tuple:
            def _actor_loss_fn(
                actor_model: FrozenDict,
                sequence: SequenceStep,
                weights: chex.Array,
            ) -> chex.Array:

                actor_policy = jax.vmap(jax.vmap(actor_model))(sequence.obs)
                log_probs = actor_policy.log_prob(sequence.action)[:, :-1]
                actor_loss = -jnp.mean(log_probs * weights)

                loss_info = {
                    "actor_loss": actor_loss,
                }

                return actor_loss, loss_info

            models, opt_states, buffer_state, key = update_state

            key, sample_key = jax.random.split(key)

            # SAMPLE SEQUENCES
            sequence_sample = buffer_sample_fn(buffer_state, sample_key)
            sequence: SequenceStep = sequence_sample.experience

            # CALCULATE WEIGHTS USING LATEST CRITIC
            v_t = jax.vmap(jax.vmap(models.critic_model))(sequence.obs)
            r_t = sequence.reward[:, :-1]
            d_t = (1 - sequence.done.astype(jnp.float32)[:, :-1]) * config.system.gamma
            advantages, _ = batch_truncated_generalized_advantage_estimation(
                r_t,
                d_t,
                config.system.gae_lambda,
                v_t,
                time_major=False,
                standardize_advantages=config.system.standardize_advantages,
                truncation_t=sequence.truncated[:, :-1],
            )
            weights = jnp.exp(advantages / config.system.beta)
            weights = jnp.minimum(weights, config.system.weight_clip)

            # CALCULATE ACTOR AND DUAL LOSS
            actor_grad_fn = jax.grad(_actor_loss_fn, argnums=(0), has_aux=True)
            actor_grads, actor_loss_info = actor_grad_fn(models.actor_model, sequence, weights)

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

            # UPDATE ACTOR MODEL AND OPTIMISER STATE
            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state
            )
            actor_new_model = optax.apply_updates(models.actor_model, actor_updates)

            # PACK NEW MODELS AND OPTIMISER STATE
            new_models = ActorCriticModels(actor_new_model, models.critic_model)
            new_opt_state = ActorCriticOptStates(actor_new_opt_state, opt_states.critic_opt_state)

            return (new_models, new_opt_state, buffer_state, key), actor_loss_info

        # We copy the models here to allow us to use the same critic for creating the target values.
        static_critic_model = models.critic_model
        update_state = (models, opt_states, buffer_state, key, static_critic_model)

        # UPDATE CRITIC STEPS
        update_state, critic_loss_info = jax.lax.scan(
            _update_critic_step, update_state, None, config.system.num_critic_steps
        )

        # We then remove static critic model from the update state.
        # Since we will be using the latest critic model for the actor update.
        models, opt_states, buffer_state, key, _ = update_state
        update_state = (models, opt_states, buffer_state, key)

        # UPDATE ACTOR STEPS
        update_state, actor_loss_info = jax.lax.scan(
            _update_actor_step, update_state, None, config.system.num_actor_steps
        )

        loss_info = {
            **actor_loss_info,
            **critic_loss_info,
        }

        models, opt_states, buffer_state, key = update_state
        learner_state = AWRLearnerState(
            models, opt_states, buffer_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: AWRLearnerState) -> AnakinExperimentOutput[AWRLearnerState]:
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
) -> Tuple[LearnerFn[AWRLearnerState], Actor, AWRLearnerState]:
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

    # Define actor_model, q_network and optimiser.
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

    critic_model_torso = hydra.utils.instantiate(
        config.network.critic_network.pre_torso, input_dim=observation_dim, key=keys[2]
    )
    critic_model_head = hydra.utils.instantiate(
        config.network.critic_network.critic_head,
        input_dim=critic_model_torso.output_dim,
        key=keys[3],
    )
    critic_model = Critic(torso=critic_model_torso, critic_head=critic_model_head)

    actor_lr = make_learning_rate(config.system.actor_lr, config, config.system.num_actor_steps)
    critic_lr = make_learning_rate(config.system.critic_lr, config, config.system.num_critic_steps)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise ptimiser states.
    actor_opt_state = actor_optim.init(actor_model)
    critic_opt_state = critic_optim.init(critic_model)

    models = ActorCriticModels(
        actor_model,
        critic_model,
    )
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)

    # Pack update functions.
    update_fns = (actor_optim.update, critic_optim.update)

    # Create replay buffer
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree.map(lambda x: x[None, ...], init_x)

    dummy_sequence_step = SequenceStep(
        obs=jax.tree.map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        truncated=jnp.zeros((), dtype=bool),
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


    # Define models to be replicated across devices and batches.
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
    init_learner_state = AWRLearnerState(
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
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
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
    best_params = unreplicate_batch_dim(learner_state.models.actor_model)
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
        act_opt_steps_per_eval = config.arch.num_updates_per_eval * config.system.num_actor_steps
        critic_opt_steps_per_eval = (
            config.arch.num_updates_per_eval * config.system.num_critic_steps
        )
        total_opt_steps_per_eval = act_opt_steps_per_eval + critic_opt_steps_per_eval
        train_metrics["actor_steps_per_second"] = act_opt_steps_per_eval / elapsed_time
        train_metrics["critic_steps_per_second"] = critic_opt_steps_per_eval / elapsed_time
        train_metrics["steps_per_second"] = total_opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.models.actor_model
        )  # Select only actor model
        key, *eval_keys = jax.random.split(key, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

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
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key, *eval_keys = jax.random.split(key, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
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
    config_name="default_ff_awr.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}AWR experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
