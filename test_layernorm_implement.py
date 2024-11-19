import jax
import jax.numpy as jnp
import flax.linen as nn

class Net(nn.Module):

    def setup(self):
        self.ln = nn.Dense(3)
        self.epsilon = 0.000001
        self.norm = nn.LayerNorm(use_bias=False, use_scale=False)

    def __call__(self, x):
        out = self.ln(x)
        # out = self.norm(out)
        m = jnp.mean(out, axis=1, keepdims=True)
        var = jnp.sqrt(jnp.sum((out - m) ** 2 / 3, axis=1, keepdims=True) + self.epsilon)
        # var = jnp.var(out, axis=1, keepdims=True)
        # var = jnp.sqrt(var + self.epsilon)
        # out = (out - m) / (var)
        out = (out - m) / var
        return out
    
if __name__ == "__main__":
    net = Net()
    net_params = net.init(jax.random.key(0), jnp.empty((1, 5)))
    out = net.apply(net_params, jnp.arange(0, 20).reshape(4,5))
    print(out)

# import torch
# from torch import nn

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.epsilon = 0.000001
#         self.ln = nn.Linear(5, 3)
#         self.norm = nn.LayerNorm(normalized_shape=(4, 3), elementwise_affine=False, bias=False)

#     def forward(self, x):
#         out = self.ln(x)
#         mout = torch.mean(out, dim=-1, keepdim=True)
#         # var = torch.sqrt(torch.sum((out - mout)**2, dim=1, keepdim=True)+self.epsilon)
#         var = torch.var(out, dim=-1, unbiased=False, keepdim=True)
#         var = torch.sqrt(var + self.epsilon)
#         out = (out - mout) / var
#         # out = self.norm(out)
#         return out
    

# if __name__ == "__main__":
#     net = Net()
#     out=net(torch.arange(0, 20, dtype=torch.float32).reshape(4, 5))
#     print(out)