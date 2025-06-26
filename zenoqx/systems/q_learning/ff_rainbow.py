import copy
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

from zenoqx.systems.q_learning.dqn_types import Transition
from zenoqx.utils.checkpointing import Checkpointer
from zenoqx.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from zenoqx.utils.loss import categorical_double_q_learning  # noqa: F401
from zenoqx.utils.multistep import batch_discounted_returns
from zenoqx.utils.training import make_learning_rate
from zenoqx.wrappers.episode_metrics import get_final_step_metrics

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import distrax
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
    Observation,
    OffPolicyLearnerState,
    OnlineAndTarget,
)
from zenoqx.evaluator import evaluator_setup, get_distribution_act_fn
from zenoqx.networks.base import FeedForwardStochasticActor as Actor
from zenoqx.utils import make_env as environments
from zenoqx.utils.logger import LogEvent, ZenoqxLogger
from zenoqx.utils.total_timestep_checker import check_total_timesteps


def get_warmup_fn(
    env: Environment,
    q_models: OnlineAndTarget,
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:
    def warmup(
        env_states: LogEnvState,
        timesteps: TimeStep,
        buffer_states: BufferState,
        keys: chex.PRNGKey,
    ) -> Tuple[LogEnvState, TimeStep, BufferState, jax.random.PRNGKey]:
        """Warmup the replay buffer with transitions from the initial random policy."""

        def _warmup_step_body(
            step_index: int, carry: Tuple[LogEnvState, TimeStep, BufferState, jax.random.PRNGKey]
        ) -> Tuple[LogEnvState, TimeStep, BufferState, jax.random.PRNGKey]:
            """Body of the warmup loop for a single step."""
            env_state, last_timestep, buffer_state, key = carry

            # SELECT ACTION (from initial policy)
            key, policy_key, noise_key = jax.random.split(key, num=3)
            batch_size = last_timestep.observation.agent_view.shape[0]
            noise_keys = jax.random.split(noise_key, batch_size)
            actor_policy, _, _ = jax.vmap(q_models.online)(last_timestep.observation, noise_keys)
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # PREPARE TRANSITION
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]
            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            # Add a time dimension and add to buffer
            # The buffer expects a trajectory, so we add a time axis of length 1.
            traj_batch = jax.tree.map(lambda x: x[:, None], transition)
            buffer_state = buffer_add_fn(buffer_state, traj_batch)

            return env_state, timestep, buffer_state, key

        # Use fori_loop to step for `warmup_steps` without creating a massive computation graph.
        # This is the correct way to implement a long, sequential loop in JAX.
        initial_carry = (env_states, timesteps, buffer_states, keys)
        env_states, timesteps, buffer_states, keys = jax.lax.fori_loop(
            0, config.system.warmup_steps, _warmup_step_body, initial_carry
        )

        return env_states, timesteps, keys, buffer_states

    batched_warmup_step: Callable = jax.vmap(
        warmup, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0), axis_name="batch"
    )

    return batched_warmup_step


def get_learner_fn(
    env: Environment,
    q_update_fn: optax.TransformUpdateFn,
    buffer_fns: Tuple[Callable, Callable, Callable],
    importance_weight_scheduler_fn: Callable,
    config: DictConfig,
) -> LearnerFn[OffPolicyLearnerState]:
    """Get the learner function."""

    buffer_add_fn, buffer_sample_fn, buffer_set_priorities = buffer_fns

    def _update_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: OffPolicyLearnerState, _: Any
        ) -> Tuple[OffPolicyLearnerState, Transition]:
            """Step the environment."""
            q_models, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key, noise_key = jax.random.split(key, num=3)

            batch_size = last_timestep.observation.agent_view.shape[0]
            noise_keys = jax.random.split(noise_key, batch_size)

            actor_policy, _, _ = jax.vmap(q_models.online)(last_timestep.observation, noise_keys)
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]

            transition = Transition(
                last_timestep.observation, action, timestep.reward, done, next_obs, info
            )

            learner_state = OffPolicyLearnerState(
                q_models, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, transition

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

            def _q_loss_fn(
                q_model: eqx.Module,
                target_q_model: eqx.Module,
                transitions: Transition,
                transition_probs: chex.Array,
                noise_key: chex.PRNGKey,
                importance_sampling_exponent: float,
            ) -> jnp.ndarray:

                batch_size = transitions.obs.agent_view.shape[0]

                keys = jax.random.split(noise_key, 3 * batch_size)

                _, q_logits_tm1, q_atoms_tm1 = jax.vmap(q_model)(transitions.obs, keys[:batch_size])
                _, q_logits_t, q_atoms_t = jax.vmap(target_q_model)(
                    transitions.next_obs, keys[batch_size : 2 * batch_size]
                )
                q_t_selector_dist, _, _ = jax.vmap(q_model)(
                    transitions.next_obs, keys[2 * batch_size :]
                )
                q_t_selector = q_t_selector_dist.preferences

                # Cast and clip rewards.
                discount = 1.0 - transitions.done.astype(jnp.float32)
                d_t = (discount * config.system.gamma).astype(jnp.float32)
                r_t = jnp.clip(
                    transitions.reward, -config.system.max_abs_reward, config.system.max_abs_reward
                ).astype(jnp.float32)
                a_tm1 = transitions.action

                batch_q_error = categorical_double_q_learning(
                    q_logits_tm1, q_atoms_tm1, a_tm1, r_t, d_t, q_logits_t, q_atoms_t, q_t_selector
                )

                # Importance weighting.
                importance_weights = (1.0 / transition_probs).astype(jnp.float32)
                importance_weights **= importance_sampling_exponent
                importance_weights /= jnp.max(importance_weights)

                # Reweight.
                q_loss = jnp.mean(importance_weights * batch_q_error)
                new_priorities = batch_q_error

                loss_info = {
                    "q_loss": q_loss,
                    "priorities": new_priorities,
                }

                return q_loss, loss_info

            models, opt_states, buffer_state, key = update_state

            key, sample_key, noise_key = jax.random.split(key, num=3)

            # SAMPLE TRANSITIONS
            transition_sample = buffer_sample_fn(buffer_state, sample_key)
            transition_sequence: Transition = transition_sample.experience
            # Extract the first and last observations.
            step_0_obs = jax.tree.map(lambda x: x[:, 0], transition_sequence).obs
            step_0_actions = transition_sequence.action[:, 0]
            step_n_obs = jax.tree.map(lambda x: x[:, -1], transition_sequence).next_obs
            # check if any of the transitions are done - this will be used to decide
            # if bootstrapping is needed
            n_step_done = jnp.any(transition_sequence.done, axis=-1)
            # Calculate the n-step rewards and select the first one.
            discounts = 1.0 - transition_sequence.done.astype(jnp.float32)
            n_step_reward = batch_discounted_returns(
                transition_sequence.reward,
                discounts * config.system.gamma,
                jnp.zeros_like(discounts),
            )[:, 0]
            transitions = Transition(
                obs=step_0_obs,
                action=step_0_actions,
                reward=n_step_reward,
                done=n_step_done,
                next_obs=step_n_obs,
                info=transition_sequence.info,
            )

            step_count = optax.tree_utils.tree_get(opt_states, "count")
            importance_sampling_exponent = importance_weight_scheduler_fn(step_count)

            # CALCULATE Q LOSS
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                models.online,
                models.target,
                transitions,
                transition_sample.probabilities,
                noise_key,
                importance_sampling_exponent,
            )

            # Update priorities in the buffer.
            updated_priorities = q_loss_info.pop("priorities")
            buffer_state = buffer_set_priorities(
                buffer_state, transition_sample.indices, updated_priorities
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="batch")
            q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")

            # UPDATE Q MODeL AND OPTIMISER STATE
            q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states)
            q_new_online_model = optax.apply_updates(models.online, q_updates)
            # Target model polyak update.
            new_target_q_model = optax.incremental_update(
                q_new_online_model, models.target, config.system.tau
            )
            q_new_model = OnlineAndTarget(q_new_online_model, new_target_q_model)

            # PACK NEW MOdel AND OPTIMISER STATE
            new_model = q_new_model
            new_opt_state = q_new_opt_state

            # PACK LOSS INFO
            loss_info = {
                **q_loss_info,
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
        """Learner function."""

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        def _fori_body(
            i: int, carry: Tuple[OffPolicyLearnerState, chex.ArrayTree, chex.ArrayTree]
        ) -> Tuple[OffPolicyLearnerState, chex.ArrayTree, chex.ArrayTree]:
            """Body of the training loop for a single update step."""
            learner_state, episode_metrics_storage, train_metrics_storage = carry

            # Run one update step.
            new_learner_state, (
                current_episode_metrics,
                current_train_metrics,
            ) = batched_update_step(learner_state, None)

            # Store the new metrics in their respective storage arrays.
            episode_metrics_storage = jax.tree.map(
                lambda storage, new: storage.at[i].set(new),
                episode_metrics_storage,
                current_episode_metrics,
            )
            train_metrics_storage = jax.tree.map(
                lambda storage, new: storage.at[i].set(new),
                train_metrics_storage,
                current_train_metrics,
            )

            return new_learner_state, episode_metrics_storage, train_metrics_storage

        # Get the shape/dtype of the metrics pytrees by running a shape evaluation.
        _, (ep_metric_struct, train_metric_struct) = jax.eval_shape(
            batched_update_step, learner_state, None
        )

        num_updates_per_eval = int(config.arch.num_updates_per_eval)

        # Initialise storage arrays for the metrics.
        episode_metrics_storage = jax.tree.map(
            lambda x: jnp.zeros((num_updates_per_eval,) + x.shape, dtype=x.dtype),
            ep_metric_struct,
        )
        train_metrics_storage = jax.tree.map(
            lambda x: jnp.zeros((num_updates_per_eval,) + x.shape, dtype=x.dtype),
            train_metric_struct,
        )

        initial_carry = (learner_state, episode_metrics_storage, train_metrics_storage)

        # Use fori_loop for the main training loop.
        final_learner_state, final_ep_metrics, final_train_metrics = jax.lax.fori_loop(
            0, num_updates_per_eval, _fori_body, initial_carry
        )

        return AnakinExperimentOutput(
            learner_state=final_learner_state,
            episode_metrics=final_ep_metrics,
            train_metrics=final_train_metrics,
        )

    return learner_fn


class EvalActorWrapper(eqx.Module):
    actor: eqx.Module

    def __init__(self, actor):
        self.actor = actor

    def __call__(self, x: Observation, key: chex.PRNGKey) -> distrax.EpsilonGreedy:
        return self.actor(x, key=key)[0]


def learner_setup(
    env: Environment, key: chex.PRNGKey, config: DictConfig
) -> Tuple[LearnerFn[OffPolicyLearnerState], EvalActorWrapper, OffPolicyLearnerState]:
    """Initialise learner_fn, model, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of observations.
    observation_dim = env.observation_spec().agent_view.shape[0]
    config.system.observation_dim = observation_dim

    # Get number of actions.
    action_dim = int(env.action_spec().num_values)
    config.system.action_dim = action_dim

    # PRNG keys.
    key, *keys = jax.random.split(key, num=4)
    # Define actor_network and optimiser.
    q_model_torso = hydra.utils.instantiate(
        config.network.actor_network.pre_torso, input_dim=observation_dim, key=keys[0]
    )

    q_model_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        input_dim=q_model_torso.output_dim,
        action_dim=action_dim,
        epsilon=config.system.training_epsilon,
        key=keys[1],
    )

    q_network = Actor(torso=q_model_torso, action_head=q_model_action_head)

    eval_q_model_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        input_dim=q_model_torso.output_dim,
        action_dim=action_dim,
        epsilon=config.system.evaluation_epsilon,
        key=keys[2],
    )

    eval_q_model = Actor(torso=q_model_torso, action_head= eval_q_model_action_head)
    eval_q_model = EvalActorWrapper(actor= eval_q_model)

    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(q_lr, eps=1e-5),
    )

    # anneal the importance sampling exponent
    importance_sampling_exponent_scheduler: Callable = optax.linear_schedule(
        init_value=config.system.importance_sampling_exponent,
        end_value=1.0,
        transition_steps=config.arch.num_updates * config.system.epochs,
        transition_begin=0,
    )

    # Initialise q model and optimiser state.
    q_opt_state = q_optim.init(eqx.filter(q_network, eqx.is_array))

    models = OnlineAndTarget(
        q_network, q_network
    )  # start with online and target networks being the same
    opt_states = q_opt_state

    update_fns = q_optim.update
    scheduler_fns = importance_sampling_exponent_scheduler

    # Create replay buffer
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree.map(lambda x: x[None, ...], init_x)
    dummy_transition = Transition(
        obs=jax.tree.map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((), dtype=int),
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

    buffer_fn = fbx.make_prioritised_trajectory_buffer(
        max_size=config.system.buffer_size,
        min_length_time_axis=config.system.n_step,
        sample_batch_size=config.system.batch_size,
        add_batch_size=config.arch.num_envs,
        sample_sequence_length=config.system.n_step,
        period=1,
        priority_exponent=config.system.priority_exponent,
        device="cpu",
    )

    buffer_fns = (buffer_fn.add, buffer_fn.sample, buffer_fn.set_priorities)
    buffer_states = buffer_fn.init(dummy_transition)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, update_fns, buffer_fns, scheduler_fns, config)
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

    def reshape_states(x: chex.Array) -> chex.Array:
        return x.reshape(
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


    # Define model to be replicated across devices and batches.
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)

    def reshape_keys(x: chex.PRNGKey) -> jax.random.PRNGKey:
        return x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])

    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))

    replicate_learner = (models, opt_states, buffer_states)

    # Duplicate learner for update_batch_size.
    def broadcast(x: chex.Array) -> chex.Array:
        return jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)

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

    return learn,  eval_q_model, init_learner_state


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
    learn,  eval_q_model, learner_state = learner_setup(env, key_l, config)

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_model, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_distribution_act_fn(config,  eval_q_model, EvalActorWrapper),
        model=learner_state.models.online,
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
    max_episode_return = jnp.float32(-1e7)
    best_model = unreplicate_batch_dim(learner_state.models.online)
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
            learner_output.learner_state.models.online
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
    config_name="default_ff_rainbow.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}Rainbow experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
