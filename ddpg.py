import copy
import torch
from torch import nn
from torch.nn import functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, max_action):
        super().__init__()
        '''
        input_dim: input dimension
        action_dim: number of action can take
        max_action: maximum range of action
        '''
        self.ln_1 = nn.Linear(input_dim, 256)
        self.ln_2 = nn.Linear(256, 256)
        self.ln_3 = nn.Linear(256, action_dim)

        self.max_action = max_action


    def forward(self, state):
        '''
        scale output to [-max_action; +max_action] by tanh func
        '''
        out = F.relu(self.ln_1(state))
        out = F.relu(self.ln_2(out))
        out = F.tanh(self.ln_3(out)) * self.max_action
        return out
    

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        '''
        with a given state and action calculate q-value
        '''
        self.ln_1 = nn.Linear(input_dim + action_dim, 256)
        self.ln_2 = nn.Linear(256, 256)
        self.ln_3 = nn.Linear(256, 1)


    def forward(self, state, action):
        combine_input = torch.cat([state, action], dim=1)
        out = F.relu(self.ln_1(combine_input))
        out = F.relu(self.ln_2(out))
        out = self.ln_3(out)
        return out
    

class DDPG(object):
    def __init__(
            self, 
            input_dim, 
            action_dim, 
            max_action,
            gamma,
            tau,
    ):
        '''
        '''
        self.actor = Actor(input_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(input_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.gamma = gamma
        self.tau = tau


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flattern()


    def train(
            self,
            replay_buffer,
            batch_size,
    ):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        # compute q value
        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (not_done * self.gamma * target_Q).detach()

        current_Q = self.critic(state, action)
        # critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor loss
        actor_loss = -self.critic_target(state, self.actor(state)).mean()
        # optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # polyak update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.copy_(param.data * self.tau + (1-self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.copy_(param.data * self.tau + (1-self.tau) * target_param.data)
 

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict, filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)