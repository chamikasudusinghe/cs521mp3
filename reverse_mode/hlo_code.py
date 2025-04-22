import jax
import jax.numpy as jnp

def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

dy_dx1 = jax.grad(f, argnums=0)
dy_dx2 = jax.grad(f, argnums=1)

hlo1 = jax.jit(dy_dx1).lower(2.0, 5.0).compiler_ir("hlo").as_hlo_text()
hlo2 = jax.jit(dy_dx2).lower(2.0, 5.0).compiler_ir("hlo").as_hlo_text()

print("hlo IR for dy_dx1")
print(hlo1)

print("\nhlo IR for dy_dx2")
print(hlo2)
