import jax
import jax.numpy as jnp

def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

# gradients
dy_dx1 = jax.grad(f, argnums=0)
dy_dx2 = jax.grad(f, argnums=1)

@jax.jit
def g(x1, x2):
    return f(x1, x2), dy_dx1(x1, x2), dy_dx2(x1, x2)

x1s = jnp.linspace(1.0, 10.0, 1000)
x2s = x1s + 1.0
g_vmap_both = jax.vmap(g, in_axes=(0, 0))
result_both = g_vmap_both(x1s, x2s)

print("vectorized over both x1s and x2s")
print("f(x):", result_both[0][:5])
print("df/dx1:", result_both[1][:5])
print("df/dx2:", result_both[2][:5])

g_vmap_x1_only = jax.vmap(g, in_axes=(0, None))
result_x1_only = g_vmap_x1_only(x1s, 0.5)

print("\nvectorized over x1s only, x2=0.5")
print("f(x):", result_x1_only[0][:5])
print("df/dx1:", result_x1_only[1][:5])
print("df/dx2:", result_x1_only[2][:5])

print("\nvectorization over both")
print(jax.make_jaxpr(g_vmap_both)(x1s, x2s))

print("\nvectorization over x1 only")
print(jax.make_jaxpr(g_vmap_x1_only)(x1s, 0.5))
