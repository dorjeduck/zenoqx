# --- This would be in your main training script ---
from 
import shutil

# Model definition from our previous example
class DynamicMLP(eqx.Module):
    layers: list
    def __init__(self, config: dict, key):
        in_size, out_size, hidden_size = config["in_size"], config["out_size"], config["hidden_size"]
        activation_fn = jax.nn.relu if config["activation_name"] == "relu" else jax.nn.gelu
        keys = jax.random.split(key, 2)
        self.layers = [eqx.nn.Linear(in_size, hidden_size, key=keys[0]), eqx.nn.Lambda(activation_fn), eqx.nn.Linear(hidden_size, out_size, key=keys[1])]
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x

@eqx.filter_jit
def make_step(state: TrainingState, x, y, optimizer):
    # (Same make_step function as before)
    def loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jnp.mean((y - pred_y) ** 2)
    loss_value, grads = eqx.filter_value_and_grad(loss)(state.model, x, y)
    updates, new_opt_state = optimizer.update(grads, state.opt_state, eqx.filter(state.model, eqx.is_array))
    new_model = eqx.apply_updates(state.model, updates)
    new_key, _ = jax.random.split(state.key)
    return TrainingState(model=new_model, opt_state=new_opt_state, step=state.step + 1, key=new_key), loss_value

def run():
    # --- Run 1: Initial Training ---
    print("\n--- RUN 1: Initial training and saving ---")
    
    # Clean up previous run
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
    
    config_v1 = {"in_size": 2, "out_size": 1, "hidden_size": 64, "activation_name": "relu"}

    # Initialize checkpointer
    checkpointer = EquinoxCheckpointer(model_name="my_dynamic_model")

    # Initialize state
    key = jax.random.PRNGKey(42)
    model_key, state_key = jax.random.split(key, 2)
    model = DynamicMLP(config_v1, key=model_key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    state = TrainingState(model=model, opt_state=opt_state, step=0, key=state_key)

    # Dummy training
    for i in range(5):
        state, _ = make_step(state, jax.random.normal(key, (32,2)), jax.random.normal(key, (32,1)), optimizer)
    print(f"Finished initial training at step {state.step}.")

    # Save the final state
    checkpointer.save(timestep=state.step, state=state, config=config_v1)
    print(f"Saved checkpoint for step {state.step}.")

    # --- Run 2: Restoration ---
    print("\n--- RUN 2: Restoring from self-contained checkpoint ---")

    # Use the same checkpointer instance (in a real app, you'd just re-instantiate it)
    # checkpointer = EquinoxCheckpointer(model_name="my_dynamic_model", checkpoint_uid="...")

    # Restore the state from disk. Note we don't need any existing state object.
    # We just need to know the Model and Optimizer classes.
    restored_state = checkpointer.restore(
        ModelClass=DynamicMLP,
        OptimizerClass=optax.adam,
    )

    print(f"✅ Successfully restored state. Resuming from step {restored_state.step}.")
    
    # Continue training
    for i in range(5):
        restored_state, _ = make_step(restored_state, jax.random.normal(key, (32,2)), jax.random.normal(key, (32,1)), optimizer)
    
    print(f"Finished continued training at step {restored_state.step}.")

if __name__ == "__main__":
    run()