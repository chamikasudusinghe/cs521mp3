import jax
import jax.numpy as jnp
import timeit

# device
device = jax.devices("gpu")[0]
# device = jax.devices("cpu")[0]
print("running on:", device)

def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

dy_dx1 = jax.grad(f, argnums=0)
dy_dx2 = jax.grad(f, argnums=1)

# g1
g1 = lambda x1, x2: (
    jax.jit(f)(x1, x2),
    jax.jit(dy_dx1)(x1, x2),
    jax.jit(dy_dx2)(x1, x2)
)

# g2
g2 = jax.jit(lambda x1, x2: (
    f(x1, x2),
    dy_dx1(x1, x2),
    dy_dx2(x1, x2)
))

x1 = jax.device_put(2.0, device)
x2 = jax.device_put(5.0, device)

# warm up
g1(x1, x2)
g2(x1, x2)

print("g1 time:", timeit.timeit(lambda: g1(x1, x2), number=1000), "sec (1000 runs)")
compiled_g1 = jax.jit(g1).lower(x1, x2).compiler_ir(dialect='hlo')
print(compiled_g1.as_hlo_text())

print("g2 time:", timeit.timeit(lambda: g2(x1, x2), number=1000), "sec (1000 runs)")
compiled_g2 = jax.jit(g2).lower(x1, x2).compiler_ir(dialect='hlo')
print(compiled_g2.as_hlo_text())

# device
# device = jax.devices("gpu")[0]
device = jax.devices("cpu")[0]
print("running on:", device)

def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

dy_dx1 = jax.grad(f, argnums=0)
dy_dx2 = jax.grad(f, argnums=1)

# g1
g1 = lambda x1, x2: (
    jax.jit(f)(x1, x2),
    jax.jit(dy_dx1)(x1, x2),
    jax.jit(dy_dx2)(x1, x2)
)

# g2
g2 = jax.jit(lambda x1, x2: (
    f(x1, x2),
    dy_dx1(x1, x2),
    dy_dx2(x1, x2)
))

x1 = jax.device_put(2.0, device)
x2 = jax.device_put(5.0, device)

# warm up
g1(x1, x2)
g2(x1, x2)

print("g1 time:", timeit.timeit(lambda: g1(x1, x2), number=1000), "sec (1000 runs)")
compiled_g1 = jax.jit(g1).lower(x1, x2).compiler_ir(dialect='hlo')
print(compiled_g1.as_hlo_text())

print("g2 time:", timeit.timeit(lambda: g2(x1, x2), number=1000), "sec (1000 runs)")
compiled_g2 = jax.jit(g2).lower(x1, x2).compiler_ir(dialect='hlo')
print(compiled_g2.as_hlo_text())