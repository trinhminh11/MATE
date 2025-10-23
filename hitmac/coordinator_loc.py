import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

# -----------------------
# Helpers
# -----------------------
def sample_from_probs(probs, deterministic=False):
    """
    probs: [N_agents, action_dim] or [batch, ...]
    returns: actions tensor (N_agents,) int64
    """
    if deterministic:
        return probs.argmax(dim=-1)
    else:
        # multinomial expects probs to sum to 1 on last dim
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


# -----------------------
# Core nets
# -----------------------
class PolicyNet(nn.Module):
    """Return logits for discrete actions (no sampling inside)."""
    def __init__(self, input_dim, action_space, device="cpu"):
        super().__init__()
        self.device = device
        num_outputs = action_space.n
        self.actor = nn.Linear(input_dim, num_outputs)
        nn.init.xavier_uniform_(self.actor.weight, gain=0.1)
        nn.init.zeros_(self.actor.bias)

    def forward(self, x):
        # x: (N_agents, input_dim)
        logits = self.actor(x)  # (N_agents, num_outputs)
        return logits  # raw logits


class ValueNet(nn.Module):
    def __init__(self, input_dim, num=1):
        super().__init__()
        self.critic = nn.Linear(input_dim, num)
        nn.init.xavier_uniform_(self.critic.weight, gain=0.1)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        # x: (1, feature_dim) or (N, feature_dim) depending usage
        return self.critic(x)


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.feature_dim = output_dim

    def forward(self, input_x):
        # input_x: (N_agents, input_dim)
        return self.layer(input_x)  # (N_agents, output_dim)


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, goals, obs):
        """
        goals: (N_agents, query_dim)
        obs:   (N_agents, key_dim)
        We'll treat batch dim = 1 for attention: expand to (1, N, dim)
        Returns: inferred_context (N_agents, hidden_dim)
        """
        Q = self.query_proj(goals).unsqueeze(0)  # (1, N, hidden)
        K = self.key_proj(obs).unsqueeze(0)      # (1, N, hidden)
        V = self.value_proj(obs).unsqueeze(0)    # (1, N, hidden)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (1, N, N)
        weights = F.softmax(scores, dim=-1)  # (1, N, N)
        inferred = torch.matmul(weights, V).squeeze(0)  # (N, hidden)
        return inferred


# -----------------------
# Mixture-of-Policies components
# -----------------------
class AlphaNet(nn.Module):
    def __init__(self, input_dim, num_skills):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_skills)
        )

    def forward(self, x):
        # x: (N_agents, input_dim)
        logits = self.net(x)
        alphas = F.softmax(logits, dim=-1)  # (N_agents, num_skills)
        return alphas


class Executor(nn.Module):
    """Local skill policy π_i(a|o_i), outputs action probabilities."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs):
        # obs: (N_agents, obs_dim)
        logits = self.net(obs)  # (N_agents, action_dim)
        probs = F.softmax(logits, dim=-1)
        return probs


class MoPNetwork(nn.Module):
    """
    Given per-agent obs and per-agent goal-features, compute alphas and form
    mixed policy: pi_C = sum_i alpha_i * pi_i(.|o).
    """
    def __init__(self, obs_dim, goal_dim, num_skills, action_dim):
        super().__init__()
        self.num_skills = num_skills
        self.action_dim = action_dim
        self.alpha_net = AlphaNet(obs_dim + goal_dim, num_skills)
        self.executors = nn.ModuleList([Executor(obs_dim, action_dim) for _ in range(num_skills)])

    def forward(self, obs, final_g):
        """
        obs: (N_agents, obs_dim)
        final_g: (N_agents, goal_dim)
        returns:
            mixed_probs: (N_agents, action_dim)
            alphas: (N_agents, num_skills)
            executors_probs: (num_skills, N_agents, action_dim)
        """
        concat = torch.cat([obs, final_g], dim=-1)  # (N, obs+goal)
        alphas = self.alpha_net(concat)  # (N, num_skills)

        # compute each executor's probs for all agents
        exec_probs = []
        for executor in self.executors:
            p = executor(obs)  # (N, action_dim)
            exec_probs.append(p.unsqueeze(0))  # (1, N, action_dim)
        exec_probs = torch.cat(exec_probs, dim=0)  # (num_skills, N, action_dim)

        # transpose to (N, num_skills, action_dim)
        exec_probs = exec_probs.permute(1, 0, 2)  # (N, num_skills, action_dim)
        alphas_exp = alphas.unsqueeze(-1)  # (N, num_skills, 1)
        weighted = exec_probs * alphas_exp  # (N, num_skills, action_dim)
        mixed = weighted.sum(dim=1)  # (N, action_dim)

        return mixed, alphas, exec_probs


# -----------------------
# Coordinator + top model
# -----------------------
class Coordinator(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.encoder = LinearEncoder(obs_space, 128)
        self.actor = PolicyNet(128, action_space)  # returns logits per agent
        self.critic = ValueNet(128)

    def forward(self, x):
        """
        x: (N_agents, obs_space)
        returns:
            obs_encoded: (N_agents, feat)
            actor_logits: (N_agents, action_dim)
            values: (1,1) or (batch,1)
        """
        obs_encoded = self.encoder(x)  # (N, feat=128)
        # critic gets a global pooled input; here mean over agents
        pooled = obs_encoded.mean(dim=0, keepdim=True)  # (1, feat)
        values = self.critic(pooled)  # (1,1)
        actor_logits = self.actor(obs_encoded)  # (N, action_dim)
        return obs_encoded, actor_logits, values


class Ours(nn.Module):
    def __init__(self, obs_space, num_skills=4, action_dim=3):
        super().__init__()
        self.num_skills = num_skills
        self.action_dim = action_dim

        self.coordinator = Coordinator(obs_space, spaces.Discrete(action_dim))
        # cross-attention: queries are small goal vectors (we'll embed discrete actions to size 4)
        self.goal_embed = nn.Embedding(action_dim, 4)
        self.cross_attention = CrossAttentionBlock(query_dim=4, key_dim=128, hidden_dim=64)

        # MLP to compute Final_G from concat(inferred_context, goal_embed)
        self.mlp = nn.Sequential(
            nn.Linear(64 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(64, 32)  # final_G dim = 32

        # MoP: uses per-agent obs (128) and per-agent final_G (32)
        self.mop = MoPNetwork(obs_dim=128, goal_dim=32, num_skills=num_skills, action_dim=action_dim)

    def forward(self, x, deterministic=False):
        """
        x: (N_agents, obs_space)
        returns dict with:
          - final_G: (N_agents, final_dim)
          - value: (1,1)
          - mixed_probs: (N_agents, action_dim)
          - alphas: (N_agents, num_skills)
          - sampled_actions: (N_agents,) int64
        """
        obs_encoded, actor_logits, values = self.coordinator(x)  # actor_logits unused for skill mixture
        # derive "goal indices" per agent from actor_logits (simple argmax)
        goal_idx = actor_logits.argmax(dim=-1)  # (N_agents,)
        goal_vec = self.goal_embed(goal_idx)    # (N_agents, 4)

        inferred_context = self.cross_attention(goal_vec, obs_encoded)  # (N_agents, 64)
        concat_G = torch.cat([inferred_context, goal_vec], dim=-1)  # (N_agents, 64+4)
        hidden = self.mlp(concat_G)  # (N_agents, 64)
        final_G = self.final_layer(hidden)  # (N_agents, 32)

        # Now mixture-of-policies
        mixed_probs, alphas, exec_probs = self.mop(obs_encoded, final_G)  # mixed_probs: (N, action_dim)

        sampled_actions = sample_from_probs(mixed_probs, deterministic=deterministic)

        return {
            "final_G": final_G,
            "value": values,
            "mixed_probs": mixed_probs,
            "alphas": alphas,
            "executor_probs": exec_probs,   # (num_skills, N, action_dim) permuted earlier
            "sampled_actions": sampled_actions
        }


# -----------------------
# Quick smoke test
# -----------------------
if __name__ == "__main__":
    # Example: 8 cameras, each obs dim = 154
    N_agents = 8
    obs_dim = 154
    x = torch.randn(N_agents, obs_dim)

    # instantiate model: 4 executors (skills), action_dim=3
    model = Ours(obs_space=obs_dim, num_skills=4, action_dim=3)
    out = model(x, deterministic=False)

    print("final_G shape:", out["final_G"].shape)        # (8, 32)
    print("value shape:", out["value"].shape)            # (1,1)
    print("mixed_probs shape:", out["mixed_probs"].shape) # (8, 3)
    print("alphas shape:", out["alphas"].shape)          # (8, 4)
    print("sampled_actions shape:", out["sampled_actions"].shape)  # (8,)
