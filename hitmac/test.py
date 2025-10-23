import mate
import sys
sys.path.append("/Users/macbook/Documents/Code/MATE_minhtrinh")
env = mate.make('MultiAgentTracking-v0')
env.seed(0)
done = True
camera_joint_observation, target_joint_observation = env.reset()
print(f"Camera shape: {camera_joint_observation.shape}")
while not done:
    camera_joint_action, target_joint_action = env.action_space.sample()  # your agent here (this takes random actions)
    (
        (camera_joint_observation, target_joint_observation),
        (camera_team_reward, target_team_reward),
        done,
        (camera_infos, target_infos)
    ) = env.step((camera_joint_action, target_joint_action))