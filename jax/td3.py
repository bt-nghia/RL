"""TODO(bt-nghia): check algo"""

import copy
import functools

import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import flax.training.train_state as train_state

@jax.jit
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
    

class TD3(object):
    def __init__(
            self, 
            input_dim, 
            action_dim, 
            max_action, 
            gamma, 
            tau,
            policy_delay,
    ):

        key, skey = jax.random.split(jax.random.key(0))
        self.actor = Actor(input_dim, action_dim, max_action)
        actor_params = self.actor.init(key, jnp.empty((1, input_dim)))
        actor_optimizer = optax.adam(3e-4)
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_optimizer,
        )
        self.actor_target_params = copy.deepcopy(actor_params)
        
        self.critic = Critic(input_dim, action_dim, max_action)
        critic1_params = self.critic.init(skey, jnp.empty((1, input_dim)), jnp.empty((1, action_dim)))
        critic1_optimizer = optax.adam(3e-4)
        self.critic1_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic1_params,
            tx=critic1_optimizer,
        )
        self.critic1_target_params = copy.deepcopy(critic1_params)

        key, skey = jax.random.split(skey)
        # self.critic2 = Critic(input_dim, action_dim, max_action)
        critic2_params = self.critic.init(skey, jnp.empty((1, input_dim)), jnp.empty((1, action_dim)))
        critic2_optimizer = optax.adam(3e-4)
        self.critic2_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic2_params,
            tx=critic2_optimizer,
        )
        self.critic2_target_params = copy.deepcopy(critic2_params)

        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.iter = 0

        del actor_params
        del critic1_params
        del critic2_params
        del actor_optimizer
        del critic1_optimizer
        del critic2_optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def policy(self, state, params):
        state = jnp.array(state).reshape(1, -1)
        return self.actor.apply(params, state).flatten()

    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss(
            self,
            state,
            action,
            next_state,
            reward,
            not_done,
            critic_params,
            critic1_target_params,
            critic2_target_params,
            actor_target_params,
    ):
        next_action = self.actor.apply(actor_target_params, next_state)
        next_q1 = jax.lax.stop_gradient(self.critic.apply(critic1_target_params, next_state, next_action))
        next_q2 = jax.lax.stop_gradient(self.critic.apply(critic2_target_params, next_state, next_action))
        
        target_q = reward + jnp.minimum(next_q1, next_q2) * self.gamma * not_done
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
            critic1_params,
            critic1_target_params,
            critic2_params,
            critic2_target_params,
            actor_params,
            actor_target_params,
    ):
        qvalue = self.critic.apply(critic1_target_params, state, self.actor.apply(actor_params, state))
        return -qvalue.mean()

    @functools.partial(jax.jit, static_argnums=0)
    def update_Q(
            self,
            state,
            action,
            next_state,
            reward,
            not_done,
            critic1_state,
            critic1_target_params,
            critic2_state,
            critic2_target_params,
            actor_state,
            actor_target_params,
    ):
        critic1_grads = jax.grad(self.critic_loss, argnums=5)(state, action, next_state, reward, not_done, critic1_state.params,
                                                            critic1_target_params, critic2_target_params, actor_target_params)
        
        critic2_grads = jax.grad(self.critic_loss, argnums=5)(state, action, next_state, reward, not_done, critic2_state.params,
                                                            critic1_target_params, critic2_target_params, actor_target_params)

        critic1_state = critic1_state.apply_gradients(grads=critic1_grads)
        critic2_state = critic2_state.apply_gradients(grads=critic2_grads)
        
        return [
            critic1_state,
            critic2_state,
        ]
    
    @functools.partial(jax.jit, static_argnums=0)
    def update_policy(
            self,
            state,
            action,
            next_state,
            reward,
            not_done,
            critic1_state,
            critic1_target_params,
            critic2_state,
            critic2_target_params,
            actor_state,
            actor_target_params,
    ):
        actor_grad = jax.grad(self.actor_loss, argnums=9)(state, action, next_state, reward, not_done, critic1_state.params, 
                                                            critic2_target_params, critic2_state.params, critic2_target_params,
                                                            actor_state.params, actor_target_params)

        actor_state = actor_state.apply_gradients(grads=actor_grad)

        # polyak update
        actor_target_params = copy.deepcopy(polyak_update(actor_state.params, actor_target_params, self.tau))
        critic1_target_params = copy.deepcopy(polyak_update(critic1_state.params, critic1_target_params, self.tau))
        critic2_target_params = copy.deepcopy(polyak_update(critic2_state.params, critic2_target_params, self.tau))

        return [
            critic1_target_params,
            critic2_target_params,
            actor_state,
            actor_target_params
        ]

    def select_action(self, state):
        return self.policy(state, self.actor_state.params)

    def train(
            self,
            replay_buffer,
            batch_size,
    ):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        self.critic1_state, self.critic2_state = \
        self.update_Q(
            state,
            action,
            next_state,
            reward,
            not_done,
            self.critic1_state,
            self.critic1_target_params,
            self.critic2_state,
            self.critic2_target_params,
            self.actor_state,
            self.actor_target_params,
        )

        if self.iter % self.policy_delay == 0:
            self.critic1_target_params, self.critic2_target_params, \
            self.actor_state, self.actor_target_params = \
            self.update_policy(
                state,
                action,
                next_state,
                reward,
                not_done,
                self.critic1_state,
                self.critic1_target_params,
                self.critic2_state,
                self.critic2_target_params,
                self.actor_state,
                self.actor_target_params,
            )
        
        self.iter+=1
                                                                                                    