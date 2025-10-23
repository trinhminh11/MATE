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
from config import *
from model import DIAYNAgent

def main():
    skill_reward_history = [[] for _ in range(NUM_SKILLS)]

    print(f"--- Stage 1: Pre-training {NUM_SKILLS} Executors for MATE ---")
    print(f"--- Using Device: {DEVICE} ---")
    
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