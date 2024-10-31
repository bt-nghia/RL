# import torch
# from torch import nn
# from torch.nn import functional as F

# class QNetwork(nn.Module):
#     def __init__(self, in_dim, action_dim):
#         super().__init__()
#         self.ln1 = nn.Linear(in_dim, 256)
#         self.act = nn.ReLU()
#         self.ln2 = nn.Linear(256, action_dim)


#     def forward(self, state):
#         out = self.ln1(state)
#         out = self.act(out)
#         out = self.ln2(out)
#         return out
    

# class DQN(object):
#     qnet: QNetwork
#     qtarget: QNetwork


#     def train(
#             self, 
#             replay_buffer,
#             gamma,
#             optimizer,
#     ):
#         st, at, rt, st_next = replay_buffer.sample()

#         with torch.no_grad():
#             next_q_value = self.qtarget(st).max(dim=1)
#             target_q_value = next_q_value * gamma + rt

#         current_q_value = torch.gather(self.qnet(st), dim=1, index=at)

#         loss = F.mse_loss(current_q_value, target_q_value)
#         loss.backward
#         optimizer.step()

#         # update target net


#     def learn(
#             self,

#     )