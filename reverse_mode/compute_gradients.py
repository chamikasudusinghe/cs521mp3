import jax
import jax.numpy as jnp

# define the function
def f(x1, x2):
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

# compute gradients using reverse-mode AD
dy_dx1 = jax.grad(f, argnums=0)
dy_dx2 = jax.grad(f, argnums=1)

# evaluate
x1_val = 2.0
x2_val = 5.0
f_val = f(x1_val, x2_val)
grad_x1 = dy_dx1(x1_val, x2_val)
grad_x2 = dy_dx2(x1_val, x2_val)

print("function f(x1, x2):", f_val)
print("gradient wrt x1:", grad_x1)
print("gradient wrt x2:", grad_x2)

# output jaxprs IR
print("\n=== jaxpr for dy_dx1 ===")
print(jax.make_jaxpr(dy_dx1)(x1_val, x2_val))

print("\n=== jaxpr for dy_dx2 ===")
print(jax.make_jaxpr(dy_dx2)(x1_val, x2_val))
