import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze
from flax.training import train_state
import optax
from resnet18_flax import ResNet18

def create_train_state(rng, model, variables):
    tx = optax.sgd(learning_rate=0.01)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)

def load_pytorch_npz(path):
    return dict(np.load(path))

def convert_params(pytorch_weights):
    params = {}
    batch_stats = {}

    def conv_name(i):
        return f"conv_{i}"

    def bn_name(i):
        return f"bn_{i}"

    conv_bn_idx = 0

    # Initial conv + bn
    params[conv_name(conv_bn_idx)] = {
        "kernel": pytorch_weights["conv1.weight"].transpose(2, 3, 1, 0)
    }
    conv_bn_idx += 1
    params[bn_name(conv_bn_idx)] = {
        "scale": pytorch_weights["bn1.weight"],
        "bias": pytorch_weights["bn1.bias"],
    }
    batch_stats[bn_name(conv_bn_idx)] = {
        "mean": pytorch_weights["bn1.mean"],
        "var": pytorch_weights["bn1.var"],
    }
    conv_bn_idx += 1

    block_map = {
        "layer1": 64,
        "layer2": 128,
        "layer3": 256,
        "layer4": 512,
    }

    for stage_id, (layer_prefix, _) in enumerate(block_map.items()):
        for block_id in [0, 1]:
            block_name = f"{layer_prefix}.{block_id}"

            for conv_id in [1, 2]:
                conv_key = f"{block_name}.conv{conv_id}.weight"
                bn_key = f"{block_name}.bn{conv_id}"
                if conv_key in pytorch_weights:
                    params[conv_name(conv_bn_idx)] = {
                        "kernel": pytorch_weights[conv_key].transpose(2, 3, 1, 0)
                    }
                    conv_bn_idx += 1

                    params[bn_name(conv_bn_idx)] = {
                        "scale": pytorch_weights[f"{bn_key}.weight"],
                        "bias": pytorch_weights[f"{bn_key}.bias"],
                    }
                    batch_stats[bn_name(conv_bn_idx)] = {
                        "mean": pytorch_weights[f"{bn_key}.mean"],
                        "var": pytorch_weights[f"{bn_key}.var"],
                    }
                    conv_bn_idx += 1

            shortcut_key = f"{block_name}.shortcut.0.weight"
            if shortcut_key in pytorch_weights:
                params[conv_name(conv_bn_idx)] = {
                    "kernel": pytorch_weights[shortcut_key].transpose(2, 3, 1, 0)
                }
                conv_bn_idx += 1

                bn_shortcut_key = f"{block_name}.shortcut.1"
                params[bn_name(conv_bn_idx)] = {
                    "scale": pytorch_weights[f"{bn_shortcut_key}.weight"],
                    "bias": pytorch_weights[f"{bn_shortcut_key}.bias"],
                }
                batch_stats[bn_name(conv_bn_idx)] = {
                    "mean": pytorch_weights[f"{bn_shortcut_key}.mean"],
                    "var": pytorch_weights[f"{bn_shortcut_key}.var"],
                }
                conv_bn_idx += 1

    params["dense_0"] = {
        "kernel": pytorch_weights["linear.weight"].T,
        "bias": pytorch_weights["linear.bias"],
    }

    print(f"mapped {conv_bn_idx} conv/bn layers + dense layer")

    return {
        "params": params,
        "batch_stats": batch_stats,
    }

def main():
    rng = jax.random.PRNGKey(0)
    model = ResNet18()

    pytorch_weights = load_pytorch_npz("resnet18_pytorch_weights.npz")
    variables = convert_params(pytorch_weights)
    state = create_train_state(rng, model, freeze(variables))

    dummy_input = jnp.ones([1, 32, 32, 3])
    logits = state.apply_fn(freeze(variables), dummy_input, mutable=False)
    print("output logits:", logits)

if __name__ == "__main__":
    main()