import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.training import train_state
import optax
from resnet18_flax import ResNet18

def create_train_state(rng, model, learning_rate=0.001):
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))  # input shape
    tx = optax.sgd(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def main():
    rng = jax.random.PRNGKey(0)
    model = ResNet18()
    state = create_train_state(rng, model)

    # test
    dummy_input = jnp.ones([1, 32, 32, 3])
    logits = state.apply_fn(state.params, dummy_input)
    print("output logits shape:", logits.shape)
    print("output logits:", logits)

if __name__ == "__main__":
    main()