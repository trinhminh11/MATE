import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
import random
from collections import deque

# --- Hyperparameters ---
ENV_NAME = "HalfCheetah-v4"
NUM_SKILLS = 50                 # Number of skills to learn (z-dimension)
HIDDEN_DIM = 256                # Hidden layer size
LEARNING_RATE = 3e-4            # Learning rate for all networks
GAMMA = 0.99                    # Discount factor
TAU = 0.005                     # Soft update coefficient
ALPHA = 0.1                     # Entropy regularization coefficient (from paper)
REPLAY_BUFFER_SIZE = 1_000_000  # Size of the replay buffer
BATCH_SIZE = 256                # Batch size for training
MAX_TIMESTEPS = 1_000_000       # Total pre-training steps
MAX_EPISODE_STEPS = 1000        # Max steps per episode
LOG_P_Z = -np.log(NUM_SKILLS)   # log p(z), constant for uniform prior

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Replay Buffer ---
class ReplayBuffer:
    """A simple replay buffer."""
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done, skill):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done, skill))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, skills = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.FloatTensor(np.array(actions)).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(DEVICE),
            torch.LongTensor(np.array(skills)).to(DEVICE) # Skills are indices
        )

    def __len__(self):
        return len(self.buffer)

# --- Networks ---

class Discriminator(nn.Module):
    """Predicts the skill z given the state s. q(z|s)"""
    def __init__(self, state_dim, num_skills, hidden_dim=HIDDEN_DIM):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills)
            # Output is logits
        )

    def forward(self, state):
        return self.network(state)

    def get_log_prob(self, state, skill):
        """Returns log q(z|s) for the given skill."""
        logits = self.forward(state)
        log_softmax = F.log_softmax(logits, dim=-1)
        # Gather the log-probability of the specific skill
        return log_softmax.gather(1, skill.unsqueeze(1))

class Actor(nn.Module):
    """Policy network pi(a|s,z). Conditioned on skill z."""
    def __init__(self, state_dim, action_dim, num_skills, hidden_dim=HIDDEN_DIM):
        super(Actor, self).__init__()
        # Input is state + one-hot skill vector
        input_dim = state_dim + num_skills
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, skill_one_hot):
        x = torch.cat([state, skill_one_hot], dim=1)
        x = self.network(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, skill_one_hot):
        mean, log_std = self.forward(state, skill_one_hot)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log-probability, correcting for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    """Q-Network Q(s,a,z). Conditioned on skill z."""
    def __init__(self, state_dim, action_dim, num_skills, hidden_dim=HIDDEN_DIM):
        super(Critic, self).__init__()
        # Input is state + one-hot skill vector + action
        input_dim = state_dim + num_skills + action_dim
        
        # Q1 network
        self.network1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network (for Twin Q-Learning, standard in SAC)
        self.network2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, skill_one_hot, action):
        x = torch.cat([state, skill_one_hot, action], dim=1)
        return self.network1(x), self.network2(x)

# --- DIAYN Agent ---

class DIAYNAgent:
    def __init__(self, state_dim, action_dim, num_skills):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_skills = num_skills

        # Create networks
        self.actor = Actor(state_dim, action_dim, num_skills).to(DEVICE)
        self.critic = Critic(state_dim, action_dim, num_skills).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim, num_skills).to(DEVICE)
        self.discriminator = Discriminator(state_dim, num_skills).to(DEVICE)

        # Copy initial weights
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    def _skill_to_one_hot(self, skill_indices):
        """Convert a tensor of skill indices to a one-hot tensor."""
        return F.one_hot(skill_indices, num_classes=self.num_skills).float()

    def calculate_pseudo_reward(self, next_state_batch, skill_batch):
        """Calculate the intrinsic reward: r_t = log q(z|s_{t+1}) - log p(z)"""
        with torch.no_grad():
            log_q_z_given_s = self.discriminator.get_log_prob(next_state_batch, skill_batch)
            # LOG_P_Z is a negative number, so we subtract it (i.e., add its absolute value)
            return log_q_z_given_s - LOG_P_Z

    def select_action(self, state, skill_idx):
        """Select an action for exploration."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        skill_tensor = torch.LongTensor([skill_idx]).to(DEVICE)
        skill_one_hot = self._skill_to_one_hot(skill_tensor)
        
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor, skill_one_hot)
        
        return action.cpu().numpy().flatten()

    def update(self, batch_size):
        """Perform one update step for all networks."""
        batch = self.replay_buffer.sample(batch_size)
        states, actions, pseudo_rewards, next_states, dones, skills = batch
        
        # Convert skill indices to one-hot vectors
        skills_one_hot = self._skill_to_one_hot(skills)

        # --- 1. Update Discriminator ---
        # Goal: Maximize log q(z|s) for the correct skill
        discriminator_logits = self.discriminator(states)
        # Standard cross-entropy loss
        discriminator_loss = F.cross_entropy(discriminator_logits, skills)
        
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # --- 2. Update Critic (Q-Networks) ---
        with torch.no_grad():
            # Get next actions and log probs from policy
            next_actions, next_log_probs = self.actor.sample(next_states, skills_one_hot)
            
            # Get target Q values
            target_q1, target_q2 = self.critic_target(next_states, skills_one_hot, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # SAC target: Q_target = r + gamma * (1 - done) * (Q_min - alpha * log_pi)
            target_q = pseudo_rewards + (1 - dones) * GAMMA * (target_q - ALPHA * next_log_probs)

        # Get current Q values
        current_q1, current_q2 = self.critic(states, skills_one_hot, actions)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 3. Update Actor (Policy) ---
        # Goal: Maximize E[Q(s,a,z) - alpha * log_pi(a|s,z)]
        new_actions, new_log_probs = self.actor.sample(states, skills_one_hot)
        
        q1, q2 = self.critic(states, skills_one_hot, new_actions)
        q = torch.min(q1, q2)
        
        actor_loss = (ALPHA * new_log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 4. Soft Update Target Networks ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
        return discriminator_loss.item(), critic_loss.item(), actor_loss.item()

    def save_models(self, filename_prefix):
        """Save the actor and discriminator models."""
        torch.save(self.actor.state_dict(), f"{filename_prefix}_actor.pth")
        torch.save(self.discriminator.state_dict(), f"{filename_prefix}_discriminator.pth")
        print(f"Models saved to {filename_prefix}_actor.pth and {filename_prefix}_discriminator.pth")

# --- Main Training Loop ---
def main():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DIAYNAgent(state_dim, action_dim, NUM_SKILLS)

    state, _ = env.reset()
    # Sample a skill z for the first episode
    current_skill = np.random.randint(NUM_SKILLS)
    episode_steps = 0
    episode_reward = 0  # Tracks pseudo-reward

    print(f"Starting pre-training for {MAX_TIMESTEPS} steps...")

    for t in range(MAX_TIMESTEPS):
        # Select action based on state and current skill
        action = agent.select_action(state, current_skill)
        
        # Take action
        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_steps += 1
        
        # Calculate DIAYN pseudo-reward
        # r_t = log q(z|s_{t+1}) - log p(z)
        pseudo_reward = agent.calculate_pseudo_reward(
            torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE),
            torch.LongTensor([current_skill]).to(DEVICE)
        ).item()
        
        episode_reward += pseudo_reward
        
        # Store in replay buffer
        # We store the *pseudo_reward*, not the environment reward
        agent.replay_buffer.add(state, action, pseudo_reward, next_state, done, current_skill)
        
        state = next_state

        # Check if episode is done
        if done or episode_steps >= MAX_EPISODE_STEPS:
            print(f"Step: {t+1}/{MAX_TIMESTEPS} | Episode Steps: {episode_steps} | Pseudo-Reward: {episode_reward:.2f} | Skill: {current_skill}")
            
            # Reset environment and sample a new skill
            state, _ = env.reset()
            current_skill = np.random.randint(NUM_SKILLS)
            episode_steps = 0
            episode_reward = 0

        # Perform training update
        if len(agent.replay_buffer) > BATCH_SIZE:
            d_loss, c_loss, a_loss = agent.update(BATCH_SIZE)
            
            if t % 1000 == 0:
                print(f"Update @ Step {t}: D_Loss={d_loss:.4f}, C_Loss={c_loss:.4f}, A_Loss={a_loss:.4f}")
    
    # End of training
    print("Pre-training complete.")
    agent.save_models("diayn_pretrain")
    env.close()

if __name__ == "__main__":
    main()
