import jax
import jax.numpy as jnp


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = jnp.zeros((max_size, state_dim))
		self.action = jnp.zeros((max_size, action_dim))
		self.next_state = jnp.zeros((max_size, state_dim))
		self.reward = jnp.zeros((max_size, 1))
		self.not_done = jnp.zeros((max_size, 1))


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = jax.random.randint(jax.random.key(0), minval=0, maxval=self.size, shape=batch_size)

		return (
			jnp.array(self.state[ind]).to(self.device),
			jnp.array(self.action[ind]).to(self.device),
			jnp.array(self.next_state[ind]).to(self.device),
			jnp.array(self.reward[ind]).to(self.device),
			jnp.array(self.not_done[ind]).to(self.device)
		)