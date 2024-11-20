import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

import copy
import numpy as np
import functools


@jax.jit
def soft_update(target_params, online_params, tau: float = 0.005):
    return jax.tree.map(lambda x, y: (1 - tau) * x + tau * y, target_params, online_params)


class Critic(nn.Module):

    def setup(self):
        self.ln1 = nn.Dense(256)
        self.ln2 = nn.Dense(256)
        self.ln3 = nn.Dense(1)

        self.ln4 = nn.Dense(256)
        self.ln5 = nn.Dense(256)
        self.ln6 = nn.Dense(1)

    def __call__(self, state, action):
        inp = jnp.concat([state, action], axis=1)
        out1 = nn.relu(self.ln1(inp))
        out1 = nn.relu(self.ln2(out1))
        out1 = self.ln3(out1)

        out2 = nn.relu(self.ln4(inp))
        out2 = nn.relu(self.ln5(out2))
        out2 = self.ln6(out2)
        return out1, out2
    

class Actor(nn.Module):
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
    

class TD3(object):
    def __init__(
        self,
        input_dim,
        action_dim,
        max_action,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        noise_clip=0.5,
        policy_noise=0.2,
    ):
        self.it = 0
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = max_action

        self.key = jax.random.key(0)
        self.key, skey = jax.random.split(self.key)
        self.actor = Actor(action_dim, max_action)
        actor_params = self.actor.init(skey, jnp.empty((1, input_dim)))
        actor_opt = optax.adam(3e-4)
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_opt,
        )

        self.key, skey = jax.random.split(self.key)
        self.critic = Critic()
        critic_params = self.critic.init(skey, jnp.empty((1, input_dim)), jnp.empty((1, action_dim)))
        critic_opt = optax.adam(3e-4)
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=critic_opt,
        )

        self.actor_target_params = copy.deepcopy(actor_params)
        self.critic_target_params = copy.deepcopy(critic_params)

        del actor_opt
        del actor_params
        del critic_opt
        del critic_params
        
    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss(
        self,
        state,
        action,
        reward,
        next_state,
        not_done,
        critic_params,
        critic_target_params,
        actor_params,
        actor_target_params,
        key,
    ):
        
        # noise = (np.random.normal(size=action.shape) * self.policy_noise).clip(-self.noise_clip, self.noise_clip) # very important
        noise = (jax.random.normal(key, shape=action.shape) * self.policy_noise).clip(-self.noise_clip, self.noise_clip)
        next_action = self.actor.apply(actor_target_params, next_state)
        next_action = (next_action + noise).clip(-self.max_action, self.max_action)
        next_q1, next_q2 = self.critic.apply(critic_target_params, next_state, next_action)
        next_q = jax.lax.min(next_q1, next_q2)
        target_q = jax.lax.stop_gradient(reward + self.gamma * not_done * next_q)

        cur_q1, cur_q2 = self.critic.apply(critic_params, state, action)
        loss = jnp.mean((cur_q1 - target_q)**2) + jnp.mean((cur_q2 - target_q)**2)
        return loss

    @functools.partial(jax.jit, static_argnums=0)
    def update_critic(
        self,
        state,
        action,
        reward,
        next_state,
        not_done,
        critic_state,
        critic_target_params,
        actor_state,
        actor_target_params,
        key,
    ):
        critic_grad = jax.grad(self.critic_loss, argnums=5)(
            state,
            action,
            reward,
            next_state,
            not_done,
            critic_state.params,
            critic_target_params,
            actor_state.params,
            actor_target_params,
            key,
        )
        critic_state = critic_state.apply_gradients(grads=critic_grad)
        return critic_state
    
    @functools.partial(jax.jit, static_argnums=0)
    def actor_loss(
        self,
        state,
        action,
        reward,
        next_state,
        not_done,
        critic_params,
        critic_target_params,
        actor_params,
        actor_target_params,
    ):
        act = self.actor.apply(actor_params, state)
        q1, q2 = self.critic.apply(critic_params, state, act)
        act_loss = -q1.mean()
        return act_loss
    
    @functools.partial(jax.jit, static_argnums=0)
    def update_actor(
        self,
        state,
        action,
        reward,
        next_state,
        not_done,
        critic_state,
        critic_target_params,
        actor_state,
        actor_target_params,
    ):
        actor_grad = jax.grad(self.actor_loss, argnums=7)(
            state,
            action,
            reward,
            next_state,
            not_done,
            critic_state.params,
            critic_target_params,
            actor_state.params,
            actor_target_params,
        )

        actor_state = actor_state.apply_gradients(grads=actor_grad)
        return actor_state
    
    @functools.partial(jax.jit, static_argnums=0)
    def policy(self, state, params):
        state = jnp.array(state).reshape(1, -1)
        return self.actor.apply(params, state).flatten()

    def select_action(self, state):
        return self.policy(state, self.actor_state.params)
    
    def train(
        self,
        replay_buffer,
        batch_size
    ):
        self.it+=1
        self.key, skey = jax.random.split(self.key)
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        self.critic_state = self.update_critic(
            state,
            action,
            reward,
            next_state,
            not_done,
            self.critic_state,
            self.critic_target_params,
            self.actor_state,
            self.actor_target_params,
            skey,
        )

        if self.it % self.policy_delay == 0:
            self.actor_state = self.update_actor(
                state,
                action,
                reward,
                next_state,
                not_done,
                self.critic_state,
                self.critic_target_params,
                self.actor_state,
                self.actor_target_params,
            )

            self.critic_target_params = soft_update(self.critic_target_params, self.critic_state.params)
            self.actor_target_params = soft_update(self.actor_target_params, self.actor_state.params)            
