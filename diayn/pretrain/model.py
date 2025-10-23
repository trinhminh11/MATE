import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque
from config import *
class RunningMeanStd:
    """Tracks running mean and variance for normalization."""
    def __init__(self, shape):
        self.mean = torch.zeros(shape, device=DEVICE)
        self.var = torch.ones(shape, device=DEVICE)
        self.count = 1e-4  # small number to prevent division by zero

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)


def init_weights_xavier(m):
    """Applies Xavier initialization to all Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
class Executor(nn.Module):
    """Continuous policy π(a|s) for 2D camera action."""
    def __init__(self, obs_dim, action_dim=2, hidden_dim=128, log_std_min=-5, log_std_max=1):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # Apply Xavier initialization
        self.apply(init_weights_xavier)

        # Small initial log_std bias to avoid saturation
        nn.init.constant_(self.log_std_layer.bias, -1.0)

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
        z = dist.rsample()  # reparameterization trick
        z = torch.clamp(z, -5, 5)  # prevents tanh saturation

        action = torch.tanh(z)
        # Correct log-prob for tanh squashing
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean


class Critic(nn.Module):
    """Twin Q-networks."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(init_weights_xavier)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class Discriminator(nn.Module):

    def __init__(self, state_dim, num_skills, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_skills)
        )

        self.apply(init_weights_xavier)

    def forward(self, state):
        return self.network(state)

    def get_log_prob(self, state, skill):
        logits = self.forward(state)
        log_softmax = F.log_softmax(logits, dim=-1)
        return log_softmax.gather(1, skill.unsqueeze(1))


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    def add(self, state, action, reward, next_state, done, skill):
        self.buffer.append((state, action, reward, next_state, done, skill))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, skills = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.FloatTensor(np.array(actions)).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(DEVICE),
            torch.LongTensor(np.array(skills)).to(DEVICE)
        )
    def __len__(self):
        return len(self.buffer)


# --- DIAYN Agent ---
class DIAYNAgent:
    def __init__(self, state_dim, action_dim, num_skills):
        self.num_skills = num_skills
        self.action_dim = action_dim
        self.state_norm = RunningMeanStd(state_dim)   # 🧠 <── NEW: track normalization stats

        self.executors = nn.ModuleList([Executor(state_dim, action_dim).to(DEVICE) for _ in range(num_skills)])
        self.critics = nn.ModuleList([Critic(state_dim, action_dim).to(DEVICE) for _ in range(num_skills)])
        self.critic_targets = nn.ModuleList([Critic(state_dim, action_dim).to(DEVICE) for _ in range(num_skills)])
        self.discriminator = Discriminator(state_dim, num_skills).to(DEVICE)

        for i in range(num_skills):
            self.critic_targets[i].load_state_dict(self.critics[i].state_dict())

        self.executor_optimizers = [optim.Adam(exec.parameters(), lr=LEARNING_RATE) for exec in self.executors]
        self.critic_optimizers = [optim.Adam(crit.parameters(), lr=LEARNING_RATE) for crit in self.critics]
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    @torch.no_grad()
    def calculate_pseudo_reward(self, next_state_batch, skill_batch):
        # Normalize observation before feeding into discriminator
        self.state_norm.update(next_state_batch)
        s_norm = self.state_norm.normalize(next_state_batch)
        log_q_z_given_s = self.discriminator.get_log_prob(s_norm, skill_batch)
        pseudo_reward = log_q_z_given_s - LOG_P_Z
        return pseudo_reward

    @torch.no_grad()
    def select_action(self, joint_obs, skill_idx):
        state_tensor = torch.FloatTensor(joint_obs).to(DEVICE)
        state_tensor = self.state_norm.normalize(state_tensor)  
        action, _, _ = self.executors[skill_idx].sample(state_tensor)
        return action.cpu().numpy()

    def update(self, batch_size):
        states, actions, pseudo_rewards_in, next_states, dones, skills = self.replay_buffer.sample(batch_size)

        self.state_norm.update(states)
        s_norm = self.state_norm.normalize(states)

        log_q_z_given_s = self.discriminator.get_log_prob(s_norm, skills)
        disc_loss = -log_q_z_given_s.mean()
        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        self.discriminator_optimizer.step()
        disc_loss_value = disc_loss.item()

        pseudo_rewards = pseudo_rewards_in
        per_skill_losses = [None] * self.num_skills

        # --- Actor-Critic update per skill ---
        for z in range(self.num_skills):
            mask = (skills == z)
            if mask.sum() == 0:
                continue
            s_z, a_z, r_z, ns_z, d_z = states[mask], actions[mask], pseudo_rewards[mask], next_states[mask], dones[mask]

            s_z = self.state_norm.normalize(s_z)
            ns_z = self.state_norm.normalize(ns_z)

            with torch.no_grad():
                next_action, next_log_prob, _ = self.executors[z].sample(ns_z)
                tq1, tq2 = self.critic_targets[z](ns_z, next_action)
                target_q = r_z + (1 - d_z) * GAMMA * (torch.min(tq1, tq2) - ALPHA * next_log_prob)

            q1, q2 = self.critics[z](s_z, a_z)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.critic_optimizers[z].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[z].step()

            # --- Actor ---
            new_action, log_prob, _ = self.executors[z].sample(s_z)
            q1_pi, q2_pi = self.critics[z](s_z, new_action)
            q_min = torch.min(q1_pi, q2_pi)
            actor_loss = (ALPHA * log_prob - q_min).mean()
            self.executor_optimizers[z].zero_grad()
            actor_loss.backward()
            self.executor_optimizers[z].step()

            # --- Soft Update ---
            for tp, p in zip(self.critic_targets[z].parameters(), self.critics[z].parameters()):
                tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)

            per_skill_losses[z] = (critic_loss.item(), actor_loss.item())

        return disc_loss_value, per_skill_losses
    
    
# cái graph thì có phải mình sẽ chia ra 2 node, 1 node là để xử lý text, node còn lại là tool calling à a.
# cái node tool calling thì sẽ gọi hoặc không tuỳ vào prompt a