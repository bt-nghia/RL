import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

class Critic(nn.Module):
    def setup(self):
        self.ln1 = nn.Dense(256)
        self.ln2 = nn.Dense(256)
        self.ln3 = nn.Dense(1)

    def __call__(self, state):
        out = nn.relu(self.ln1(state))
        out = nn.relu(self.ln2(out))
        out = self.ln3(out)
        return out
    

class Actor(nn.Module):
    dim: int
    max_action: int

    def setup(self):
        self.ln1 = nn.Dense(256)
        self.ln2 = nn.Dense(256)
        self.ln3 = nn.Dense(1)


    def __call__(self, state, action):
        inp = jnp.concat([state, action], axis=1)
        out = nn.relu(self.ln1(inp))
        out = nn.relu(self.ln2(out))
        out = nn.tanh(self.ln3(out)) * self.max_action
        return out
    

class SAC(object):
    def __init__(
        self,
        dim,
        max_action,
    ):
        pass
