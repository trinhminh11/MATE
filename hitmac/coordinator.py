import torch
import torch.nn as nn
import torch.nn.functional as F
from perception import AttentionLayer
from torch.autograd import Variable
from gym import spaces

'''
Original:
    Camera observation: (8, 154)

'''
def sample_action(mu_multi, sigma_multi, device = "cpu", test=False):
    # discrete
    logit = mu_multi
    prob = F.softmax(logit, dim=-1)
    log_prob = F.log_softmax(logit, dim=-1)
    entropy = -(log_prob * prob).sum(-1, keepdim=True)
    if test:
        action = prob.max(-1)[1].data
        action_env = action.cpu().numpy()  # np.squeeze(action.cpu().numpy(), axis=0)
    else:
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))  # [num_agent, 1] # comment for sl slave
        action_env = action.squeeze(0)

    return action_env, entropy, log_prob


class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, device = "cpu"):
        super().__init__()
        self.device = device
        num_outputs = action_space.n

        self.actor = nn.Linear(input_dim, num_outputs)
        # simple layer init
        nn.init.xavier_uniform_(self.actor.weight, gain=0.1)
        self.actor.bias.data.zero_()

    def forward(self, x, test=False):
        logits = F.relu(self.actor(x))
        sigma = torch.ones_like(logits)
        action, entropy, log_prob = sample_action(logits, sigma, self.device, test)
        return action, entropy, log_prob


class ValueNet(nn.Module):
    def __init__(self, input_dim, num=1):
        super().__init__()

        self.critic = nn.Linear(input_dim, num)
        nn.init.xavier_uniform_(self.critic.weight, gain=0.1)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        return self.critic(x)


            
class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim = 32):
        super(LinearEncoder, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )     
        self.feature_dim = output_dim
        
    def forward(self, input_x):
        return self.layer(input_x)

class CrossAttentionBlock(nn.Module):
    """
    Fuses Goal features (G) with Observation features (O) to create Inferred Context (I).
    Uses G as Queries and O as Keys/Values (or vice versa, the forward method defines it).
    """
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Linear layers for Query, Key, Value projection
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)

        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

    def forward(self, goals, obs):
        """
        Calculates attention where Queries are Goals (G) and Keys/Values are Observations (O).
        Input shapes are (Batch, N_Agents, Dim).
        """
        Q = self.query_proj(goals)  # Queries: Goals (G)
        K = self.key_proj(obs)      # Keys: Observations (O)
        V = self.value_proj(obs)    # Values: Observations (O)

        # 1. Attention Scores: (Batch, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 2. Attention Weights
        weights = F.softmax(scores, dim=-1)

        # 3. Inferred Context (I): (Batch, N, Hidden_Dim)
        inferred_context = torch.matmul(weights, V)
        
        # We assume I has the same structure as G/O for concatenation later.
        return inferred_context 


class Coordinator(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.encoder = LinearEncoder(obs_space, 128)
        feature_dim = self.encoder.feature_dim
        self.actor = PolicyNet(feature_dim, action_space)
        self.critic = ValueNet(feature_dim)

    def forward(self, x):
        obs_encoded = self.encoder(x)
        values = self.critic(obs_encoded.mean(dim=0, keepdim=True))
        actions, entropies, log_probs = self.actor(obs_encoded)
        return obs_encoded, actions, values


class Ours(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.coordinator = Coordinator(obs_space, spaces.Discrete(2))
        self.cross_attention = CrossAttentionBlock(query_dim=2, key_dim=128, hidden_dim=64)
        self.goal_embed = nn.Embedding(2, 2)
        self.mlp = nn.Sequential(
            nn.Linear(64 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.final_layer = nn.Linear(64, 32)  # Final G

    def forward(self, x):
        obs_encoded, actions, values = self.coordinator(x)
        goals = self.goal_embed(actions.squeeze())
        inferred_context = self.cross_attention(goals.unsqueeze(0), obs_encoded.unsqueeze(0))
        inferred_context = inferred_context.squeeze(0)
        concat_G = torch.cat([inferred_context, goals], dim=-1)
        hidden = self.mlp(concat_G)
        final_G = self.final_layer(hidden)
        
        concat_OG = torch.cat([obs_encoded, final_G], dim=-1)
        alpha_init = self.alpha_net(concat_OG)
        return final_G, values
    

class AlphaNet(nn.Module):
    def __init__(self, input_dim, num_skills):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_skills),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)  # α_1, α_2, ..., α_N

class Executor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs):
        return self.policy(obs)  # π_i(a | o_i)


class MoPNetwork(nn.Module):  # Mixture of Policies (π^C)
    def __init__(self, obs_dim, goal_dim, num_skills, action_dim):
        super().__init__()
        self.alpha = AlphaNet(obs_dim + goal_dim, num_skills)
        self.executors = nn.ModuleList([Executor(obs_dim, action_dim) for _ in range(num_skills)])

    def forward(self, obs, goals):
        concat_input = torch.cat([obs, goals], dim=-1)
        alphas = self.alpha(concat_input)  # (B, num_skills)
        
        # Each executor produces π_i(a|o)
        skill_outputs = torch.stack([π(obs) for π in self.executors], dim=-1)  # (B, action_dim, num_skills)

        # Weighted combination
        mixed_action = torch.sum(skill_outputs * alphas.unsqueeze(1), dim=-1)
        return mixed_action, alphas
    
if __name__ == '__main__':
    x = torch.zeros(8, 154)
    model = Ours(obs_space=x.size(1))
    I, V = model(x)
    print("Inferred context shape:", I.shape)
    print("Value shape:", V.shape)
    