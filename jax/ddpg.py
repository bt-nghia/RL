import copy
import functools

import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import flax.training.train_state as train_state


def mse_loss(a, b):
    return jnp.mean(jnp.square(a-b))

@jax.jit
def polyak_update(src, tgt, tau):
    params = jax.tree.map(
        lambda src, tgt: src * tau + tgt * (1.0-tau),
        src,
        tgt
    )
    return params


class Actor(nn.Module):
    input_dim: int
    action_dim: int
    max_action: float

    def setup(self):
        self.ln1 = nn.Dense(256)
        self.ln2 = nn.Dense(256)
        self.ln3 = nn.Dense(self.action_dim)

    def __call__(self, state):
        out = nn.relu(self.ln1(state))
        out = nn.relu(self.ln2(out))
        out = nn.tanh(self.ln3(out)) * self.max_action
        return out
    

class Critic(nn.Module):
    input_dim: int
    action_dim: int
    max_action: float

    def setup(self):
        self.ln1 = nn.Dense(256)
        self.ln2 = nn.Dense(256)
        self.ln3 = nn.Dense(1)

    def __call__(self, state, action):
        inp = jnp.concat([state, action], axis=1)
        out = nn.relu(self.ln1(inp))
        out = nn.relu(self.ln2(out))
        out = self.ln3(out)
        return out
    

class DDPG(object):
    def __init__(
            self, 
            input_dim, 
            action_dim, 
            max_action, 
            gamma, 
            tau
    ):
        self.actor = Actor(input_dim, action_dim, max_action)
        actor_params = self.actor.init(jax.random.key(0), jnp.empty((1, input_dim)))
        self.actor_optimizer = optax.adam(3e-4)
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=self.actor_optimizer,
        )
        self.actor_target_params = copy.deepcopy(actor_params)

        self.critic = Critic(input_dim, action_dim, max_action)
        critic_params = self.critic.init(jax.random.key(1), jnp.empty((1, input_dim)), jnp.empty((1, action_dim)))
        self.critic_optimizer = optax.adam(3e-4)
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=self.critic_optimizer,
        )
        self.critic_target_params = copy.deepcopy(critic_params)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = jnp.array(state).reshape(1, -1)
        return self.actor.apply(self.actor_state.params, state).flatten()

    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss(
            self,
            state,
            action,
            next_state,
            reward,
            not_done,
            critic_params,
            critic_target_params,
            actor_params,
            actor_target_params,
    ):
        next_q = self.critic.apply(critic_target_params, next_state, self.actor.apply(actor_target_params, next_state))
        target_q = reward + next_q * self.gamma * not_done
        current_q = self.critic.apply(critic_params, state, action)
        loss = mse_loss(current_q, target_q)
        return loss
    
    @functools.partial(jax.jit, static_argnums=0)
    def actor_loss(
            self,
            state,
            action,
            next_state,
            reward,
            not_done,
            critic_params,
            critic_target_params,
            actor_params,
            actor_target_params,
    ):
        qvalue = self.critic.apply(critic_target_params, state, self.actor.apply(actor_params, state))
        return -qvalue.mean()

    def train(
            self,
            replay_buffer,
            batch_size,
    ):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        critic_grad = jax.grad(self.critic_loss, argnums=5)(state, action, next_state, reward, not_done, self.critic_state.params, self.critic_target_params, self.actor_state.params, self.actor_target_params)
        actor_grad = jax.grad(self.actor_loss, argnums=7)(state, action, next_state, reward, not_done, self.critic_state.params, self.critic_target_params, self.actor_state.params, self.actor_target_params)

        self.critic_state = self.critic_state.apply_gradients(grads=critic_grad)
        self.actor_state = self.actor_state.apply_gradients(grads=actor_grad)

        # polyak update
        self.actor_target_params = copy.deepcopy(polyak_update(self.actor_state.params, self.actor_target_params, self.tau))
        self.critic_target_params = copy.deepcopy(polyak_update(self.critic_state.params, self.critic_target_params, self.tau))

# if __name__ == "__main__":
#     DDPG(10, 10, 10, 0.99, 0.99)