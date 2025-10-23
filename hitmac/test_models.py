import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same Executor class again
class Executor(nn.Module):
    def __init__(self, obs_dim, action_dim=2, hidden_dim=128, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.log_std_layer.bias, -2.0)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        return action


# Load executors
NUM_SKILLS = 4
OBS_DIM = 154
ACTION_DIM = 2

executors = nn.ModuleList([Executor(OBS_DIM, ACTION_DIM).to(DEVICE) for _ in range(NUM_SKILLS)])
state_dict = torch.load("diayn_executor_library_mate.pth", map_location=DEVICE)
executors.load_state_dict(state_dict)
executors.eval()
print(len(executors.state_dict()))
print("✅ Loaded DIAYN executors successfully.")
