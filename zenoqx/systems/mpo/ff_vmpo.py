import copy
import time
from typing import Any, Dict, Tuple

import chex
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
import rlax
from colorama import Fore, Style
from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from zenoqx.base_types import (
    AnakinExperimentOutput,
    LearnerFn,
    OnlineAndTarget,
)
from zenoqx.evaluator import evaluator_setup, get_distribution_act_fn
from zenoqx.networks.base import FeedForwardActor as Actor
from zenoqx.networks.base import FeedForwardCritic as Critic
from zenoqx.systems.mpo.continuous_loss import _MPO_FLOAT_EPSILON
from zenoqx.systems.mpo.discrete_loss import (
    clip_categorical_mpo_params,
    get_temperature_from_params,
)
from zenoqx.systems.mpo.mpo_types import (
    CategoricalDualParams,
    SequenceStep,
    VMPOLearnerState,
    VMPOOptStates,
    VMPOModels,
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
    batch_truncated_generalized_advantage_estimation,
)
from zenoqx.utils.total_timestep_checker import check_total_timesteps
from zenoqx.utils.training import make_learning_rate
from zenoqx.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: Environment,
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[VMPOLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic models.

    actor_update_fn, critic_update_fn, dual_update_fn = update_fns

    def _update_step(learner_state: VMPOLearnerState, _: Any) -> Tuple[VMPOLearnerState, Tuple]:
        def _env_step(
            learner_state: VMPOLearnerState, _: Any
        ) -> Tuple[VMPOLearnerState, SequenceStep]:
            """Step the environment."""
            models, opt_states, key, env_state, last_timestep, learner_step_count = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            # We act with target model in VMPO
            actor_policy = jax.vmap(models.actor_models.target)(last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)
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

            learner_state = VMPOLearnerState(
                models, opt_states, key, env_state, timestep, learner_step_count
            )
            return learner_state, sequence_step

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        models, opt_states, key, env_state, last_timestep, learner_step_count = learner_state

        # Swap the batch and time axes for easier processing.
        traj_batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        chex.assert_tree_shape_prefix(
            traj_batch, (config.arch.num_envs, config.system.rollout_length)
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the model for a single epoch."""

            def _actor_loss_fn(
                online_actor_model: eqx.Module,
                dual_params: CategoricalDualParams,
                target_actor_model: eqx.Module,
                advantages: chex.Array,
                sequence: SequenceStep,
            ) -> chex.Array:

                # Remove the last timestep from the sequence.
                sequence = jax.tree.map(lambda x: x[:, :-1], sequence)

                # Reshape the sequence to [B*T, ...].
                (sequence, advantages) = jax.tree.map(
                    lambda x: merge_leading_dims(x, 2), (sequence, advantages)
                )

                temperature = get_temperature_from_params(dual_params).squeeze()
                alpha = jax.nn.softplus(dual_params.log_alpha).squeeze() + _MPO_FLOAT_EPSILON

                online_actor_policy = jax.vmap(online_actor_model)(sequence.obs)
                target_actor_policy = jax.vmap(target_actor_model)(sequence.obs)

                sample_log_probs = online_actor_policy.log_prob(sequence.action)
                temperature_constraint = rlax.LagrangePenalty(
                    temperature, config.system.epsilon, False
                )
                kl = target_actor_policy.kl_divergence(online_actor_policy)
                alpha_constraint = rlax.LagrangePenalty(alpha, config.system.epsilon_policy, False)
                kl_constraints = [(kl, alpha_constraint)]
                # restarting_weights = 1-sequence.done.astype(jnp.float32)

                loss, loss_info = rlax.vmpo_loss(
                    sample_log_probs=sample_log_probs,
                    advantages=advantages,
                    temperature_constraint=temperature_constraint,
                    kl_constraints=kl_constraints,
                    # restarting_weights=restarting_weights
                )

                loss_info = loss_info._asdict()
                loss_info["temperature"] = temperature
                loss_info["alpha"] = alpha
                loss_info["advantages"] = advantages

                return jnp.mean(loss), loss_info

            def _critic_loss_fn(
                online_critic_model: eqx.Module,
                value_target: chex.Array,
                sequence: SequenceStep,
            ) -> chex.Array:

                # Remove the last timestep from the sequence.
                sequence = jax.tree.map(lambda x: x[:, :-1], sequence)

                online_v_t = jax.vmap(jax.vmap(online_critic_model))(sequence.obs)  # [B, T]

                td_error = value_target - online_v_t

                v_loss = rlax.l2_loss(td_error).mean()

                loss_info = {
                    "v_loss": v_loss,
                }

                return v_loss, loss_info

            models, opt_states, key, sequence_batch, learner_step_count = update_state

            # Calculate Advantages and Value Target of pi_target
            discount = 1.0 - sequence_batch.done.astype(jnp.float32)
            d_t = (discount * config.system.gamma).astype(jnp.float32)
            r_t = jnp.clip(
                sequence_batch.reward, -config.system.max_abs_reward, config.system.max_abs_reward
            ).astype(jnp.float32)

            online_v_t = jax.vmap(jax.vmap(models.critic_model))(sequence_batch.obs)  # [B, T]

            # We recompute the targets using the latest critic every time
            if config.system.use_n_step_bootstrap:
                value_target = batch_n_step_bootstrapped_returns(
                    r_t[:, :-1],
                    d_t[:, :-1],
                    online_v_t[:, 1:],
                    config.system.n_step_for_sequence_bootstrap,
                )
                advantages = value_target - online_v_t[:, :-1]
            else:
                advantages, value_target = batch_truncated_generalized_advantage_estimation(
                    r_t[:, :-1],
                    d_t[:, :-1],
                    config.system.gae_lambda,
                    online_v_t,
                    time_major=False,
                    truncation_t=sequence_batch.truncated[:, :-1],
                )

            # CALCULATE ACTOR AND DUAL LOSS
            actor_dual_grad_fn = jax.grad(_actor_loss_fn, argnums=(0, 1), has_aux=True)
            actor_dual_grads, actor_dual_loss_info = actor_dual_grad_fn(
                models.actor_models.online,
                models.dual_params,
                models.actor_models.target,
                advantages,
                sequence_batch,
            )

            # CALCULATE CRITIC LOSS
            critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
            critic_grads, critic_loss_info = critic_grad_fn(
                models.critic_model,
                value_target,
                sequence_batch,
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

            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="batch"
            )
            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="device"
            )

            actor_grads, dual_grads = actor_dual_grads

            # UPDATE ACTOR PARAMS AND OPTIMISER STATE
            actor_updates, actor_new_opt_state = actor_update_fn(
                actor_grads, opt_states.actor_opt_state
            )
            actor_new_online_model = optax.apply_updates(models.actor_models.online, actor_updates)

            # UPDATE DUAL MODEL AND OPTIMISER STATE
            dual_updates, dual_new_opt_state = dual_update_fn(dual_grads, opt_states.dual_opt_state)
            dual_new_params = optax.apply_updates(models.dual_params, dual_updates)
            dual_new_params = clip_categorical_mpo_params(dual_new_params)

            # UPDATE CRITIC PARAMS AND OPTIMISER STATE
            critic_updates, critic_new_opt_state = critic_update_fn(
                critic_grads, opt_states.critic_opt_state
            )
            critic_new_online_model = optax.apply_updates(models.critic_model, critic_updates)

            learner_step_count += 1

            # POLYAK UPDATE FOR ACTOR
            new_target_actor_model = optax.periodic_update(
                actor_new_online_model,
                models.actor_models.target,
                learner_step_count,
                config.system.actor_target_period,
            )

            # PACK NEW MODELS AND OPTIMISER STATE
            actor_new_models = OnlineAndTarget(actor_new_online_model, new_target_actor_model)

            new_models = VMPOModels(actor_new_models, critic_new_online_model, dual_new_params)
            new_opt_state = VMPOOptStates(
                actor_new_opt_state, critic_new_opt_state, dual_new_opt_state
            )

            # PACK LOSS INFO
            loss_info = {
                **actor_dual_loss_info,
                **critic_loss_info,
            }
            return (new_models, new_opt_state, key, sequence_batch, learner_step_count), loss_info

        update_state = (models, opt_states, key, traj_batch, learner_step_count)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        models, opt_states, key, traj_batch, learner_step_count = update_state
        learner_state = VMPOLearnerState(
            models, opt_states, key, env_state, last_timestep, learner_step_count
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: VMPOLearnerState) -> AnakinExperimentOutput[VMPOLearnerState]:
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
) -> Tuple[LearnerFn[VMPOLearnerState], Actor, VMPOLearnerState]:
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

    # Define actor_model, critic_model and optimiser.
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

    actor_lr = make_learning_rate(config.system.actor_lr, config, config.system.epochs)
    critic_lr = make_learning_rate(config.system.critic_lr, config, config.system.epochs)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise optimiser states.
    actor_opt_state = actor_optim.init(actor_model)
    critic_opt_state = critic_optim.init(critic_model)

    # Initialise VMPO Dual params and optimiser state.
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

    models = VMPOModels(
        OnlineAndTarget(actor_model, actor_model),
        critic_model,
        dual_params,
    )
    opt_states = VMPOOptStates(actor_opt_state, critic_opt_state, dual_opt_state)

    # Pack apply and update functions.
    update_fns = (actor_optim.update, critic_optim.update, dual_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

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
    key, step_key = jax.random.split(key, num=2)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    learner_step_count = jnp.int32(0)

    replicate_learner = (models, opt_states, learner_step_count)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = jax.device_put_replicated(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    models, opt_states, learner_step_count = replicate_learner

    init_learner_state = VMPOLearnerState(
        models, opt_states, step_keys, env_states, timesteps, learner_step_count
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
        trained_model = unreplicate_batch_dim(
            learner_output.learner_state.models.actor_models.online
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
    config_name="default_ff_vmpo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}V-MPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
