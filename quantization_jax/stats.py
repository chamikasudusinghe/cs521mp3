import jax
import jax.numpy as jnp
import numpy as np
import time
import os
from flax.core import freeze
from resnet18_jax import ResNet18
from load_pytorch_weights import convert_params, load_pytorch_npz
from quantize_jax import create_train_state, run_inference, inject_into_template

def count_param_bytes(tree):
    return sum(v.size * v.dtype.itemsize for v in jax.tree_util.tree_leaves(tree)) / 1e6  # MB

def save_raw_flat_npy(params, filename):
    flat = jax.tree_util.tree_leaves(params)
    array = np.concatenate([v.reshape(-1).astype(np.int8 if v.dtype == jnp.int8 else np.float32) for v in flat])
    np.save(filename, array)
    return os.path.getsize(filename + ".npy") / 1e6

def quantize_to_int8(params):
    scale = 127.0
    quantized = {}
    for k, v in params.items():
        if isinstance(v, dict):
            quantized[k] = quantize_to_int8(v)
        elif isinstance(v, jnp.ndarray):
            max_val = jnp.max(jnp.abs(v)) + 1e-8
            q = jnp.round((v / max_val) * scale)
            quantized[k] = q.astype(jnp.float32)
        else:
            quantized[k] = v
    return quantized

def main():
    rng = jax.random.PRNGKey(0)
    model = ResNet18()
    input_tensor = jnp.ones([1, 32, 32, 3])

    pytorch_weights = load_pytorch_npz("resnet18_pytorch_weights.npz")
    flax_converted = convert_params(pytorch_weights)
    template = model.init(rng, input_tensor)
    fp32_vars = inject_into_template(template, flax_converted)
    state_fp32 = create_train_state(rng, model, fp32_vars)

    qparams = quantize_to_int8(fp32_vars["params"])
    qparams = jax.tree_map(lambda x: x.astype(jnp.int8) if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32 else x, qparams)

    quant_vars = freeze({
        "params": qparams,
        "batch_stats": fp32_vars["batch_stats"]
    })
    state_q = create_train_state(rng, model, quant_vars)

    raw_fp32 = count_param_bytes(fp32_vars["params"])
    raw_quant = count_param_bytes(qparams)

    saved_fp32 = save_raw_flat_npy(fp32_vars["params"], "fp32_model_raw")
    saved_quant = save_raw_flat_npy(qparams, "quantized_model_raw")

    print(f"fp32 size: {raw_fp32:.2f} mb")
    print(f"int8 size: {raw_quant:.2f} mb")

    dtypes = set(v.dtype for v in jax.tree_util.tree_leaves(qparams))

    # warmup
    run_inference(state_fp32, fp32_vars, input_tensor)
    run_inference(state_q, quant_vars, input_tensor)

    def time_model(state, vars_):
      start = time.time()
      for _ in range(30):
          run_inference(state, vars_, input_tensor)
      end = time.time()
      return (end - start) * 1000

    t_fp32 = time_model(state_fp32, fp32_vars)/30
    t_quant = time_model(state_q, quant_vars)/30

    print(f"fp32 inference time (): {t_fp32:.2f} ms")
    print(f"quantized inference time: {t_quant:.2f} ms")

if __name__ == "__main__":
    main()
