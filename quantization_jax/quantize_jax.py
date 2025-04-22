import jax
import flax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.training import train_state
import optax
from resnet18_flax import ResNet18
from load_pytorch_weights import convert_params, load_pytorch_npz

def create_train_state(rng, model, variables):
    tx = optax.sgd(learning_rate=0.01)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)

def quantize_params(params, bits=8):
    scale = 2 ** bits - 1
    quantized = {}

    for k, v in params.items():
        if isinstance(v, (dict, flax.core.FrozenDict)):
            quantized[k] = quantize_params(v, bits)
        elif isinstance(v, jnp.ndarray):
            max_val = jnp.max(jnp.abs(v)) + 1e-8  # avoid zero division
            q = jnp.round((v / max_val) * scale)
            quantized[k] = (q / scale) * max_val
        else:
            quantized[k] = v
    return quantized

def run_inference(state, variables, input_tensor):
    return state.apply_fn(variables, input_tensor, mutable=False)

def inject_into_template(template, pytorch_converted):
    injected = unfreeze(template)
    for coll in ["params", "batch_stats"]:
        for k, v in pytorch_converted[coll].items():
            if k in injected[coll]:
                for param_key, param_val in v.items():
                    if param_key in injected[coll][k]:
                        injected[coll][k][param_key] = jnp.asarray(param_val)
    return freeze(injected)

def main():
    rng = jax.random.PRNGKey(0)
    model = ResNet18()
    input_tensor = jnp.ones([1, 32, 32, 3])

    template_vars = model.init(rng, input_tensor)

    # convertion weights
    pytorch_weights = load_pytorch_npz("resnet18_pytorch_weights.npz")
    flax_converted = convert_params(pytorch_weights)
    full_vars = inject_into_template(template_vars, flax_converted)

    # fp32
    state_fp32 = create_train_state(rng, model, full_vars)
    logits_fp32 = run_inference(state_fp32, full_vars, input_tensor)
    print("fp32 logits:", logits_fp32)

    # quantized
    qparams = quantize_params(full_vars["params"], bits=8)
    quantized_vars = freeze({
        "params": qparams,
        "batch_stats": full_vars["batch_stats"],
    })
    state_q = create_train_state(rng, model, quantized_vars)
    logits_q = run_inference(state_q, quantized_vars, input_tensor)
    print("quantized logits:", logits_q)

    hlo_fp32 = jax.jit(state_fp32.apply_fn).lower(full_vars, input_tensor).compiler_ir("hlo")
    hlo_q = jax.jit(state_q.apply_fn).lower(quantized_vars, input_tensor).compiler_ir("hlo")

    with open("hlo_fp32.txt", "w") as f:
        f.write(hlo_fp32.as_hlo_text())
    with open("hlo_quantized.txt", "w") as f:
        f.write(hlo_q.as_hlo_text())

if __name__ == "__main__":
    main()