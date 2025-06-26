import os
import json
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Type

import absl.logging as absl_logging
import orbax.checkpoint as ocp
from chex import Numeric
from omegaconf import DictConfig, OmegaConf

import equinox as eqx
import jax
import optax

# --- Our Established Equinox Helper Types and Functions ---

class TrainingState(eqx.Module):
    """A container for all state that needs to be saved and restored."""
    model: eqx.Module
    opt_state: optax.OptState
    step: int
    key: jax.random.PRNGKey

def get_saveable_state(state: TrainingState) -> TrainingState:
    """Filters the state to make it serializable for Orbax."""
    model_params, _ = eqx.partition(state.model, eqx.is_array)
    return eqx.tree_at(lambda s: s.model, state, model_params)


# --- The Refactored Checkpointer Class for Equinox ---

class EquinoxCheckpointer:
    """
    Model checkpointer for saving and restoring an Equinox TrainingState,
    including the model, optimizer state, and other metadata.
    """

    CHECKPOINTER_VERSION = 1.0

    def __init__(
        self,
        model_name: str,
        metadata: Optional[Dict] = None,
        rel_dir: str = "checkpoints",
        checkpoint_uid: Optional[str] = None,
        save_interval_steps: int = 1,
        max_to_keep: Optional[int] = 1,
    ):
        """Initialise the checkpointer tool.

        Args:
            model_name (str): Name of the model to be saved.
            metadata (Optional[Dict], optional): General metadata to save. Defaults to None.
            rel_dir (str, optional): Relative directory for checkpoints. Defaults to "checkpoints".
            checkpoint_uid (Optional[str], optional): Unique ID for the checkpoint run.
                If not given, a timestamp is used.
            save_interval_steps (int, optional): The interval at which checkpoints should be saved.
            max_to_keep (Optional[int], optional): Maximum number of checkpoints to keep.
        """
        warnings.filterwarnings("ignore", category=UserWarning, message="Couldn't find sharding info")
        absl_logging.set_verbosity(absl_logging.WARNING)

        checkpoint_str = checkpoint_uid if checkpoint_uid else datetime.now().strftime("%Y%m%d%H%M%S")
        self.directory = os.path.join(os.getcwd(), rel_dir, model_name, checkpoint_str)

        if metadata is not None and isinstance(metadata, DictConfig):
            metadata = OmegaConf.to_container(metadata, resolve=True)

        # The modern, explicit Orbax API for handling multiple item types.
        self._manager = ocp.CheckpointManager(
            self.directory,
            item_handlers={
                'train_state': ocp.PyTreeCheckpointHandler(),
                'config': ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(
                save_interval_steps=save_interval_steps,
                max_to_keep=max_to_keep,
                create=True
            ),
            metadata={
                "checkpointer_version": self.CHECKPOINTER_VERSION,
                **(metadata if metadata is not None else {}),
            },
        )

    def save(
        self,
        timestep: int,
        state: TrainingState,
        config: Dict,
    ) -> bool:
        """Saves the Equinox training state and its architectural config.

        Args:
            timestep (int): The current training step.
            state (TrainingState): The full Equinox TrainingState.
            config (Dict): The model's architectural config, to be saved as JSON.

        Returns:
            bool: Whether the save was successful.
        """
        # Partition the model to get the saveable parameters
        saveable_state = get_saveable_state(state)
        
        # Use Composite to save multiple, named items with their specific handlers
        save_args = ocp.args.Composite(
            train_state=ocp.args.PyTreeSave(item=saveable_state),
            config=ocp.args.JsonSave(config)
        )
        
        return self._manager.save(step=timestep, args=save_args)

    def restore(
        self,
        ModelClass: Type[eqx.Module],
        OptimizerClass: Type[optax.GradientTransformation],
        timestep: Optional[int] = None,
    ) -> TrainingState:
        """
        Restores a self-contained checkpoint from disk.
        It first restores the config, then uses it to build a template model
        for restoring the full training state.

        Args:
            ModelClass (Type[eqx.Module]): The class of the model to be instantiated
                (e.g., DynamicMLP).
            OptimizerClass (Type[optax.GradientTransformation]): The optimizer class
                used during training (e.g. optax.adam).
            timestep (Optional[int], optional): Specific step to restore.
                Defaults to the latest available.

        Returns:
            TrainingState: The fully restored and rehydrated training state.
        """
        step_to_restore = timestep if timestep is not None else self._manager.latest_step()
        if step_to_restore is None:
            raise FileNotFoundError(f"No checkpoint found in {self.directory}")

        # 1. Restore the config first to determine the model architecture.
        restored_items = self._manager.restore(step_to_restore, items={'config': {}})
        config = restored_items['config']
        
        # 2. Build a template model and state using the restored config.
        # The parameters of this template are irrelevant; they will be overwritten.
        key = jax.random.PRNGKey(0)
        model_key, state_key = jax.random.split(key, 2)
        template_model = ModelClass(config, key=model_key)
        optimizer = OptimizerClass(1e-3) # LR is stored in opt_state, so this is just for init
        template_opt_state = optimizer.init(eqx.filter(template_model, eqx.is_array))
        template_state = TrainingState(model=template_model, opt_state=template_opt_state, step=0, key=state_key)
        saveable_template = get_saveable_state(template_state)

        # 3. Restore the actual training state using the correctly-structured template.
        restored_items = self._manager.restore(
            step_to_restore, items={'train_state': saveable_template}
        )
        restored_saveable_state = restored_items['train_state']
        
        # 4. Rehydrate the full model by combining loaded params with static structure.
        _, static_model_structure = eqx.partition(template_model, eqx.is_array)
        restored_model = eqx.combine(restored_saveable_state.model, static_model_structure)

        # 5. Return the final, complete training state.
        return TrainingState(
            model=restored_model,
            opt_state=restored_saveable_state.opt_state,
            step=restored_saveable_state.step,
            key=restored_saveable_state.key
        )

    def get_cfg(self) -> DictConfig:
        """Return the metadata of the checkpoint."""
        return DictConfig(self._manager.metadata())