import jax
import jax.numpy as jnp
import flax.linen as nn


def loss(x, y):
    return -((x + y)**2).mean()

t1 = jnp.ones((2, 3))
t2 = jnp.ones((2, 3)) * 2

grads = jax.grad(loss, argnums=[0, 1])(t1, t2)

print(grads)