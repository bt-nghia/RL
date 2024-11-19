import jax.numpy as jnp
import time
import jax

start_time = time.time()


x = jnp.arange(0, 100).reshape(-1, 10)
y = jnp.arange(0, 1000).reshape(10, -1)
# z = x.dot(y)
z = x @ y

end_time = time.time()
print(end_time-start_time)