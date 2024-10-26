import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.nn import init

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions for initialization
def he_normal_init():
    """He normal initialization for weights."""
    return nn.init.kaiming_normal_

def orthogonal_init(scale=1.0):
    """Orthogonal initialization for weights."""
    def init(tensor):
        nn.init.orthogonal_(tensor, gain=scale)
    return init

class Dense(nn.Module):
    def __init__(self, out_features, kernel_init=orthogonal_init(), activation=None):
        super(Dense, self).__init__()
        self.out_features = out_features
        self.activation = activation

        # The weight and bias will be initialized later in the forward method
        self.weight = None
        self.bias = None

        self.kernel_init = kernel_init

    def forward(self, x):
        # Dynamically set in_features based on the input
        in_features = x.size(-1)

        # Initialize weight and bias only once
        if self.weight is None:
            self.weight = nn.Parameter(torch.empty((in_features, self.out_features)))
            self.kernel_init(self.weight)

        if self.bias is None:
            self.bias = nn.Parameter(torch.empty(self.out_features))
            nn.init.zeros_(self.bias)

        # Perform the linear transformation
        output = x @ self.weight + self.bias

        # Apply activation if specified
        if self.activation is not None:
            output = self.activation(output)

        return output

class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, dtype=torch.float32, device="cpu"):
        super(MLPBlock, self).__init__()

        self.fc1 = Dense(hidden_dim, kernel_init=orthogonal_init(np.sqrt(2.0)))
        self.fc2 = Dense(hidden_dim, kernel_init=orthogonal_init(np.sqrt(2.0)))
        self.dtype = dtype
        self.device = device

    def forward(self, x):
        x = x.to(DEVICE)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dtype=torch.float32, device="cpu"):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = Dense(hidden_dim * 4, kernel_init=he_normal_init())
        self.fc2 = Dense(hidden_dim, kernel_init=he_normal_init())
        self.dtype = dtype
       
    def forward(self, x):
        res = x.to(DEVICE)
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return res + x

class TanhPolicy(nn.Module):
    def __init__(self, action_dim, kernel_init_scale=1.0, device="cpu"):
        super(TanhPolicy, self).__init__()
        self.fc = Dense(action_dim, kernel_init=he_normal_init())
        self.device = device

    def forward(self, inputs):
        inputs = inputs.to(DEVICE)
        actions = self.fc(inputs)
        return torch.tanh(actions)


class LinearCritic(nn.Module):
    def __init__(self, kernel_init_scale=1.0, dtype=torch.float32, device="cpu"):
        super(LinearCritic, self).__init__()
        self.fc = Dense(1, kernel_init=orthogonal_init(kernel_init_scale)).to(device)
        self.dtype = dtype
        self.device = device

    def forward(self, inputs):
        inputs = inputs.to(DEVICE)
        inputs = inputs
        value = self.fc(inputs)
        return value

class DDPGEncoder(nn.Module):
    def __init__(self, block_type, num_blocks, hidden_dim, dtype=torch.float32, device="cpu"):
        super(DDPGEncoder, self).__init__()
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device

        if self.block_type == "mlp":
            self.encoder = MLPBlock(self.hidden_dim, dtype=self.dtype).to(self.device)

        elif self.block_type == "residual":
            layers = [Dense(self.hidden_dim).to(self.device)]
            for _ in range(self.num_blocks):
                layers.append(ResidualBlock(self.hidden_dim, dtype=self.dtype).to(self.device))
            layers.append(nn.LayerNorm(self.hidden_dim).to(self.device))
            self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder(x)
        return x

class DDPGActor(nn.Module):
    def __init__(self, block_type, num_blocks, hidden_dim, action_dim, dtype=torch.float32, device="cpu"):
        super(DDPGActor, self).__init__()
        self.device = device
        self.encoder =  DDPGEncoder(block_type, num_blocks, hidden_dim, dtype=dtype, device=self.device)
        self.predictor = TanhPolicy(action_dim, device=self.device)

    def forward(self, observations):
        observations = observations.to(self.device)
        z = self.encoder(observations)
        action = self.predictor(z)
        return action


class DDPGCritic(nn.Module):
    def __init__(self, block_type, num_blocks, hidden_dim, dtype=torch.float32, device="cpu"):
        super(DDPGCritic, self).__init__()
        self.device = device
        self.encoder = DDPGEncoder(block_type, num_blocks, hidden_dim, dtype=dtype, device=self.device)
        self.predictor = LinearCritic(device=self.device)

    def forward(self, observations, actions):
        observations, actions = observations.to(self.device), actions.to(self.device)
        inputs = torch.cat([observations, actions], dim=1)
        z = self.encoder(inputs)
        q = self.predictor(z)
        return q


class DDPGClippedDoubleCritic(nn.Module):
    def __init__(self, block_type, num_blocks, hidden_dim, dtype=torch.float32, num_qs=2, device="cpu"):
        super(DDPGClippedDoubleCritic, self).__init__()
        self.num_qs = num_qs
        self.critics = nn.ModuleList([DDPGCritic(block_type, num_blocks, hidden_dim, dtype=dtype, device=device) for _ in range(self.num_qs)])
        self.device = device

    def forward(self, observations, actions):
        observations, actions = observations.to(self.device), actions.to(self.device)
        qs = torch.stack([critic(observations, actions) for critic in self.critics], dim=0)
        return qs
