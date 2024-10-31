import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state
import copy

if __name__ == "__main__":
    x = nn.Dense(7, kernel_init=nn.initializers.ones_init(), bias_init=nn.initializers.ones_init())
    tau = 0.89
    x_params = x.init(jax.random.key(0), jnp.empty((1, 10)))
    xtx = optax.adam(3e-4)

    y = nn.Dense(7, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.zeros_init())
    y_params = y.init(jax.random.key(1), jnp.empty((1, 10)))
    yty = optax.adam(3e-4)

    x_state = train_state.TrainState.create(
        apply_fn=x.apply,
        params=x_params,
        tx=xtx
    )

    y_state = train_state.TrainState.create(
        apply_fn=y.apply,
        params=y_params,
        tx=yty
    )

    g = jax.tree_util.tree_map(lambda src, tgt: src * tau + tgt * (1.0-tau), x_params, y_params)
    
    # y_params = optax.incremental_update(
    #     new_tensors=x_params,
    #     old_tensors=y_params,
    #     step_size=tau
    # )
    
    print(y_params)
    y_params = copy.deepcopy(g)
    print(y_params)