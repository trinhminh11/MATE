import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
import random
from collections import deque
import mate
from mate.agents import GreedyTargetAgent
import matplotlib.pyplot as plt

# --- Hyperparameters ---
ENV_CONFIG = "MATE-8v8-9.yaml"
NUM_AGENTS = 8
OBS_DIM = 154
ACTION_DIM = 2
NUM_SKILLS = 4
HIDDEN_DIM = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.1
REPLAY_BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
MAX_TIMESTEPS = 1_000_000
MAX_EPISODE_STEPS = 200
LOG_P_Z = -np.log(NUM_SKILLS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROTATION_MAX = 30.0
ZOOM_MAX = 0.5
ACTION_SCALE = torch.tensor([ROTATION_MAX, ZOOM_MAX], device=DEVICE)

skill_reward_history = [[] for _ in range(NUM_SKILLS)]

print(f"--- Stage 1: Pre-training {NUM_SKILLS} Executors for MATE ---")
print(f"--- Using Device: {DEVICE} ---")
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
            
# --- Networks ---
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

        # Apply Xavier initialization
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



def main():
    base_env = gym.make('MultiAgentTracking-v0', config=ENV_CONFIG)
    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    print(f"Environment Loaded: {NUM_AGENTS} agents, ObsDim={OBS_DIM}, ActDim={ACTION_DIM}")
    agent = DIAYNAgent(OBS_DIM, ACTION_DIM, NUM_SKILLS)

    obs, _ = env.reset()
    current_skill = np.random.randint(NUM_SKILLS)
    episode_steps, total_steps, update_steps = 0, 0, 0

    disc_loss_history = []
    critic_loss_history = [[] for _ in range(NUM_SKILLS)]
    actor_loss_history = [[] for _ in range(NUM_SKILLS)]

    print(f"Starting pre-training for {MAX_TIMESTEPS} total agent steps...")

    while total_steps < MAX_TIMESTEPS:
        if total_steps % 500 == 0:  # more frequent skill switch
            current_skill = np.random.randint(NUM_SKILLS)

        actions_norm = agent.select_action(obs, current_skill)
        actions = np.clip(actions_norm * ACTION_SCALE.cpu().numpy(),
                          [-ROTATION_MAX, -ZOOM_MAX], [ROTATION_MAX, ZOOM_MAX])

        if total_steps % 2000 == 0:
            print(f"Step {total_steps:,}: sample action {actions[0]} (skill={current_skill})")

        next_obs, _, done, trunc, _ = env.step(actions)
        skill_batch = torch.LongTensor([current_skill] * NUM_AGENTS).to(DEVICE)
        pseudo_rewards = agent.calculate_pseudo_reward(torch.FloatTensor(next_obs).to(DEVICE), skill_batch)
        pseudo_rewards_np = pseudo_rewards.cpu().numpy().flatten()

        skill_reward_history[current_skill].append(np.mean(pseudo_rewards_np))

        if total_steps % 5000 == 0:
            print(f"[Skill {current_skill}] mean pseudo-reward: {np.mean(pseudo_rewards_np):.3f}")

        done_flag = done or trunc
        for i in range(NUM_AGENTS):
            agent.replay_buffer.add(obs[i], actions[i], pseudo_rewards_np[i], next_obs[i], done_flag, current_skill)

        obs = next_obs
        episode_steps += 1
        total_steps += NUM_AGENTS

        if done_flag or episode_steps >= MAX_EPISODE_STEPS:
            obs, _ = env.reset()
            episode_steps = 0

        # === TRAINING ===
        if len(agent.replay_buffer) > BATCH_SIZE * 10:
            # update() returns loss diagnostics now
            disc_loss_value, per_skill_losses = agent.update(BATCH_SIZE)
            disc_loss_history.append(disc_loss_value)
            for z in range(NUM_SKILLS):
                if per_skill_losses[z] is not None:
                    c_loss, a_loss = per_skill_losses[z]
                    critic_loss_history[z].append(c_loss)
                    actor_loss_history[z].append(a_loss)
            update_steps += 1

            if update_steps % 100 == 0:
                print(f"[Update {update_steps}] DiscLoss={disc_loss_value:.4f}, "
                      f"Replay={len(agent.replay_buffer):,}, Step={total_steps:,}")

    print("🎯 Pre-training complete.")
    agent.save_models("diayn_executor_library_mate_v2.pth")

    # === PLOTTING ===
    plt.figure(figsize=(8, 5))
    for z in range(NUM_SKILLS):
        plt.plot(skill_reward_history[z], label=f"Skill {z}")
    plt.xlabel("Training updates (~5000 steps per point)")
    plt.ylabel("Mean Pseudo-Reward")
    plt.title("DIAYN Skill Pseudo-Reward Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("diayn_skill_reward_curve.png")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(disc_loss_history)
    plt.xlabel("Training updates")
    plt.ylabel("Discriminator Loss")
    plt.title("DIAYN Discriminator Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("diayn_discriminator_loss_curve.png")
    plt.show()

    env.close()



if __name__ == "__main__":
    main()
