import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

import copy
import functools


@jax.jit
def soft_update(target_params, online_params, tau: float = 0.005):
    return jax.tree_multimap(lambda x, y: (1 - tau) * x + tau * y, target_params, online_params)



class Critic(nn.Module):

    def setup(self):
        self.ln1 = nn.Dense(256)
        self.ln2 = nn.Dense(256)
        self.ln3 = nn.Dense(1)

        self.ln4 = nn.Dense(256)
        self.ln5 = nn.Dense(256)
        self.ln6 = nn.Dense(1)        

    def __call__(self, state):
        out1 = nn.relu(self.ln1(state))
        out1 = nn.relu(self.ln2(out1))
        out1 = self.ln3(out1)

        out2 = nn.relu(self.ln4(state))
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

    def __call__(self, state, action):
        inp = jnp.concat([state, action], axis=1)
        out = nn.relu(self.ln1(inp))
        out = nn.relu(self.ln2(out))
        out = nn.tanh(self.ln3(out)) * self.max_action
        return out
    

class TD3_2(object):
    def __init__(
            self,
            input_dim,
            action_dim,
            max_action,
            gamma,
            tau,
            policy_delay=2,
    ):
        self.iter = 0
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay

        self.key = jax.random.key(0)
        self.key, skey = jax.random.split(self.key)
        self.actor = Actor(action_dim, max_action)
        actor_params = self.actor.init(skey, jnp.empty((1, input_dim)), jnp.empty((1, action_dim)))
        actor_opt = optax.adam(3e-4)
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_opt,
        )

        self.key, skey = jax.random.split(self.key)
        self.critic = Critic()
        critic_params = self.critic.init(skey, jnp.empty((1, input_dim)))
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
        
    @functools.partial(jax.jit, static_argnums=1)
    def critic_loss(
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
        next_action = jax.lax.stop_gradient(self.actor.apply(actor_target_params, next_state))
        next_q1, next_q2 = jax.lax.stop_gradient(self.critic.apply(critic_target_params, next_state, next_action))
        target_q = reward + self.gamma * not_done * jnp.minimum(next_q1, next_q2)

        cur_q1, cur_q2 = self.critic.apply(critic_state.params, state, action)
        loss = (cur_q1 - target_q)**2 + (cur_q2 - target_q)**2
        loss = loss.mean()
        return loss

    @functools.partial(jax.jit, static_argnums=1)
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
    ):
        critic_grad = jax.grad(self.critic_loss, argnums=5)(
            state,
            action,
            reward,
            next_state,
            not_done,
            critic_state,
            critic_target_params,
            actor_state,
            actor_target_params,
        )
        critic_state = critic_state.apply_gradient(grads=critic_grad)
        return critic_state
    
    @functools.partial(jax.jit, static_argnums=0)
    def actor_loss(
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
        act = self.actor.apply(actor_state.params, state)
        q1, q2 = self.critic.apply(critic_state.params, state, act)
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
        actor_grad = jax.grad(self.actor_loss, argnums=7) (
            state,
            action,
            reward,
            next_state,
            not_done,
            critic_state,
            critic_target_params,
            actor_state,
            actor_target_params,
        )

        actor_state = actor_state.apply_gradient(grads=actor_grad)
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
        self.iter+=1
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

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
        )

        if self.iter % self.policy_delay:
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

if __name__ == "__main__":
    algo = TD3_2(10, 10, 1, 0.99, 0.99, 2)