import copy
import time
from typing import Any, Dict, Tuple

import chex
import distrax
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style

from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from zenoqx.base_types import (
    ActorCriticOptStates,
    ActorCriticModels,
    AnakinExperimentOutput,
    LearnerFn,
    OnPolicyLearnerState,
)
from zenoqx.evaluator import evaluator_setup, get_distribution_act_fn
from zenoqx.networks.base import FeedForwardActor as Actor
from zenoqx.networks.base import FeedForwardCritic as Critic
from zenoqx.systems.ppo.ppo_types import PPOTransition
from zenoqx.utils import make_env as environments
from zenoqx.utils.checkpointing import Checkpointer
from zenoqx.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from zenoqx.utils.logger import LogEvent, ZenoqxLogger
from zenoqx.utils.loss import clipped_value_loss, dpo_loss
from zenoqx.utils.multistep import batch_truncated_generalized_advantage_estimation
from zenoqx.utils.total_timestep_checker import check_total_timesteps
from zenoqx.utils.training import make_learning_rate
from zenoqx.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: Environment,
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[OnPolicyLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic models.
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: OnPolicyLearnerState, _: Any
    ) -> Tuple[OnPolicyLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            learner_state (NamedTuple):
                - model (ActorCriticModel): The current model.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.
        """

        def _env_step(
            learner_state: OnPolicyLearnerState, _: Any
        ) -> Tuple[OnPolicyLearnerState, PPOTransition]:
            """Step the environment."""
            models, opt_states, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = jax.vmap(models.actor_model)(last_timestep.observation)
            value = jax.vmap(models.critic_model)(last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            transition = PPOTransition(
                done,
                truncated,
                action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                info,
            )
            learner_state = OnPolicyLearnerState(models, opt_states, key, env_state, timestep)
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        models, opt_states, key, env_state, last_timestep = learner_state
        last_val = jax.vmap(models.critic_model)(last_timestep.observation)

        r_t = traj_batch.reward
        v_t = jnp.concatenate([traj_batch.value, last_val[None, ...]], axis=0)
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)
        advantages, targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            d_t,
            config.system.gae_lambda,
            v_t,
            time_major=True,
            standardize_advantages=config.system.standardize_advantages,
            truncation_t=traj_batch.truncated,
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the model for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the model for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                models, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_model: eqx.Module,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN MODEL
                    actor_policy = jax.vmap(actor_model)(traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    def _kl_calculator(
                        b_pi: distrax.Distribution, pi: distrax.Distribution
                    ) -> chex.Array:
                        """KL divergence for continuous policies."""
                        # This is the workaround for the distrax bug.
                        # We "peel the onion" to get to the base Normal distributions.
                        kl_per_action_dim = b_pi.distribution.distribution.kl_divergence(
                            pi.distribution.distribution
                        )
                        # Sum over action dimension to get one KL value per batch item.
                        return kl_per_action_dim.sum(axis=-1)

                    # CALCULATE ACTOR LOSS
                    loss_actor = dpo_loss(
                        log_prob, traj_batch.log_prob, gae, config.system.alpha, config.system.beta
                    )
                    # CORRECT: Call entropy on the top-level Independent distribution.
                    entropy = actor_policy.distribution.entropy(seed=key).mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    loss_info = {
                        "actor_loss": loss_actor,
                        "entropy": entropy,
                    }
                    return total_loss_actor, loss_info

                '''
                def _actor_loss_fn(
                    actor_model: eqx.Module,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN MODEL
                    actor_policy = jax.vmap(actor_model)(traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # CALCULATE ACTOR LOSS
                    loss_actor = ppo_clip_loss(
                        log_prob, traj_batch.log_prob, gae, config.system.clip_eps
                    )
                    entropy = actor_policy.distribution.entropy(seed=key).mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    loss_info = {
                        "actor_loss": loss_actor,
                        "entropy": entropy,
                    }
                    return total_loss_actor, loss_info
                '''

                def _critic_loss_fn(
                    critic_model: eqx.Module,
                    traj_batch: PPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN MODEL
                    value = jax.vmap(critic_model)(traj_batch.obs)

                    # CALCULATE VALUE LOSS
                    value_loss = clipped_value_loss(
                        value, traj_batch.value, targets, config.system.clip_eps
                    )

                    critic_total_loss = config.system.vf_coef * value_loss
                    loss_info = {
                        "value_loss": value_loss,
                    }
                    return critic_total_loss, loss_info

                # CALCULATE ACTOR LOSS
                key, actor_loss_key = jax.random.split(key)
                actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(
                    models.actor_model, traj_batch, advantages, actor_loss_key
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(
                    models.critic_model, traj_batch, targets
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

                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="device"
                )

                # UPDATE ACTOR MODEL AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_model = optax.apply_updates(models.actor_model, actor_updates)

                # UPDATE CRITIC MODEL AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_model = optax.apply_updates(models.critic_model, critic_updates)

                # PACK NEW MODEL AND OPTIMISER STATE
                new_model = ActorCriticModels(actor_new_model, critic_new_model)
                new_opt_state = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

                # PACK LOSS INFO
                loss_info = {
                    **actor_loss_info,
                    **critic_loss_info,
                }
                return (new_model, new_opt_state, key), loss_info

            models, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (models, opt_states, key), loss_info = jax.lax.scan(
                _update_minibatch, (models, opt_states, key), minibatches
            )

            update_state = (models, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (models, opt_states, traj_batch, advantages, targets, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        models, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = OnPolicyLearnerState(models, opt_states, key, env_state, last_timestep)
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OnPolicyLearnerState,
    ) -> AnakinExperimentOutput[OnPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the model
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - models (ActorCriticModels): The initial models.
                - opt_states (OptStates): The initial optimizer state.
                - key (jax.random.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
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
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[OnPolicyLearnerState], Actor, OnPolicyLearnerState]:
    """Initialise learner_fn, model, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of observations.
    observation_dim = env.observation_spec().agent_view.shape[0]
    config.system.observation_dim = observation_dim

    # Get number/dimension of actions.
    num_actions = int(env.action_spec().shape[-1])
    config.system.action_dim = num_actions
    config.system.action_minimum = float(env.action_spec().minimum)
    config.system.action_maximum = float(env.action_spec().maximum)

    # PRNG keys.
    key, *keys = jax.random.split(jax.random.PRNGKey(config.arch.seed), 5)

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(
        config.network.actor_network.pre_torso, input_dim=observation_dim, key=keys[0]
    )
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        input_dim=actor_torso.output_dim,
        action_dim=num_actions,
        minimum=config.system.action_minimum,
        maximum=config.system.action_maximum,
        key=keys[1],
    )
    critic_torso = hydra.utils.instantiate(
        config.network.critic_network.pre_torso, input_dim=observation_dim, key=keys[2]
    )
    critic_head = hydra.utils.instantiate(
        config.network.critic_network.critic_head, input_dim=critic_torso.output_dim, key=keys[3]
    )

    actor_model = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_model = Critic(torso=critic_torso, critic_head=critic_head)

    actor_lr = make_learning_rate(
        config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches
    )

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree.map(lambda x: x[None, ...], init_x)

    # Initialise optimiser states.

    actor_opt_state = actor_optim.init(actor_model)
    critic_opt_state = critic_optim.init(critic_model)

    # Pack models.
    models = ActorCriticModels(actor_model, critic_model)

    # Pack apply and update functions.
    update_fns = (actor_optim.update, critic_optim.update)

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
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (models, opt_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = jax.device_put_replicated(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    models, opt_states = replicate_learner
    init_learner_state = OnPolicyLearnerState(models, opt_states, step_keys, env_states, timesteps)

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
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    # Setup learner.
    learn, actor_model, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

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
    max_episode_return = jnp.float32(-1e7)
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
        opt_steps_per_eval = config.arch.num_updates_per_eval * (
            config.system.epochs * config.system.num_minibatches
        )
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_model = unreplicate_batch_dim(
            learner_output.learner_state.models.actor_model
        )  # Select only actor model
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
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
            # Save checkpoint of learner state
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

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
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
    config_path="../../../configs/default/anakin",
    config_name="default_ff_dpo_continuous.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}DPO (continuous) experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
