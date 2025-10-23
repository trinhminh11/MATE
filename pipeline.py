import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import random
from collections import deque
import mate
from mate.agents import GreedyTargetAgent
from hitmac.coordinator_loc import DIAYNAgent

ENV_CONFIG = "MATE-8v8-9.yaml"  # Your MATE environment config
NUM_AGENTS = 8
OBS_DIM = 154
ACTION_DIM = 3                 # Discrete actions (Stay, Left, Right)
NUM_SKILLS = 4                 # Number of "expert" skills to learn
HIDDEN_DIM = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.1                     # Entropy regularization
REPLAY_BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
MAX_TIMESTEPS = 1_000_000       # Total *per-agent* steps
MAX_EPISODE_STEPS = 1000        # Max steps per episode
LOG_P_Z = -np.log(NUM_SKILLS)   # log p(z)
MODEL_SAVE_PATH = "diayn_executor_library_mate.pth" # The final output file

def main():
    base_env = gym.make('MultiAgentTracking-v0', config=ENV_CONFIG, render_mode=None) # No render for training
    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    # Verify dims
    assert env.unwrapped.num_cameras == NUM_AGENTS
    assert env.observation_space.shape[1] == OBS_DIM
    assert env.action_space.shape[1].n == ACTION_DIM
    print(f"Environment Loaded: {NUM_AGENTS} agents, ObsDim={OBS_DIM}, ActDim={ACTION_DIM}")

    agent = DIAYNAgent(OBS_DIM, ACTION_DIM, NUM_SKILLS)
    
    camera_joint_observation, _ = env.reset()
    current_skill = np.random.randint(NUM_SKILLS) # One skill for the whole team
    episode_steps = 0
    total_steps = 0

    print(f"Starting pre-training for {MAX_TIMESTEPS} total agent steps...")
    
    while total_steps < MAX_TIMESTEPS:
        # Select action for all 8 agents based on the *same* skill
        camera_joint_action = agent.select_action(camera_joint_observation, current_skill)
        
        # Step the environment
        results = env.step(camera_joint_action)
        next_camera_joint_observation, target_team_reward, done, truncated, _ = results

        # Calculate DIAYN pseudo-reward (per-agent)
        # We need to create a skill batch: (N_agents,)
        skill_batch_tensor = torch.LongTensor([current_skill] * NUM_AGENTS).to(DEVICE)
        pseudo_rewards_tensor = agent.calculate_pseudo_reward(
            torch.FloatTensor(next_camera_joint_observation).to(DEVICE),
            skill_batch_tensor
        )
        pseudo_rewards = pseudo_rewards_tensor.cpu().numpy().flatten() # (N_agents,)
        
        done_flag = done or truncated
        
        # Store one transition for *each* agent
        for i in range(NUM_AGENTS):
            agent.replay_buffer.add(
                camera_joint_observation[i], 
                camera_joint_action[i], 
                pseudo_rewards[i], # Store log_q(z|s')
                next_camera_joint_observation[i], 
                done_flag, 
                current_skill
            )
        
        camera_joint_observation = next_camera_joint_observation
        episode_steps += 1
        total_steps += NUM_AGENTS

        if done_flag or episode_steps >= MAX_EPISODE_STEPS:
            if total_steps > 0:
                print(f"Step: {total_steps}/{MAX_TIMESTEPS} | Episode Steps: {episode_steps} | Skill: {current_skill}")
            camera_joint_observation, _ = env.reset()
            current_skill = np.random.randint(NUM_SKILLS)
            episode_steps = 0

        if len(agent.replay_buffer) > BATCH_SIZE * 10:
            agent.update(BATCH_SIZE)
    
    print("Pre-training complete.")
    agent.save_models(MODEL_SAVE_PATH)
    env.close()

if __name__ == "__main__":
    main()
