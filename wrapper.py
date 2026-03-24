from typing import TypedDict

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces

import mate
from agents import GreedyCameraAgent as GreedyCameraAgent
from mate.agents import GreedyTargetAgent
from mate.constants import TERRAIN_SIZE
from mate.entities import Obstacle
from mate.wrappers import MultiCamera

from agents.base import GoalsBaseAgent


class RefinedObs(TypedDict):
    warehouses: np.ndarray
    cameras: np.ndarray
    targets: np.ndarray
    obstacles: np.ndarray

class MateCameraDictObs:
    def __init__(
        self,
        env,
        warehouse_include_R=False,
        target_include_R=False,
        target_include_loaded=False,
        expand_warehouses_and_obstacle=False,
    ):
        """
        env: the mate environment to wrap
        warehouse_include_R: whether to include the radius of the warehouse in the warehouses state
        target_include_R: whether to include the radius of the target in the target state
        target_include_loaded: whether to include the loaded flag of the target in the target state
        expand_warehouses_and_obstacle: whether to expand the warehouses and obstacle state to have a separate entry for each camera, or to keep them shared among all cameras (if False, the warehouses and obstacle state will be the same for all cameras)
        """
        self.env = env
        self.target_include_R = target_include_R
        self.target_include_loaded = target_include_loaded
        self.expand_warehouses_and_obstacle = expand_warehouses_and_obstacle
        self.warehouse_include_R = warehouse_include_R

        self.num_cameras = self.env.unwrapped.num_cameras

        warehouses_shape = self.env.unwrapped.num_warehouses * 2 + (
            1 if warehouse_include_R else 0
        )
        warehouses_shape = (
            (self.num_cameras, warehouses_shape)
            if expand_warehouses_and_obstacle
            else (warehouses_shape,)
        )

        warehouses_space = spaces.Box(
            low=-1.0, high=1.0, shape=warehouses_shape, dtype=np.float64
        )
        self_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_cameras, 8), dtype=np.float64
        )  # x, y, R, phi, theta, Rmax, phimax, thetamax

        target_dim = 3
        if target_include_R:
            target_dim += 1
        if target_include_loaded:
            target_dim += 1
        target_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(
                self.num_cameras,
                self.env.unwrapped.num_targets * target_dim,
            ),
            dtype=np.float64,
        )

        obstacle_shape = self.env.unwrapped.num_obstacles * 3
        obstacle_shape = (
            (self.num_cameras, obstacle_shape)
            if expand_warehouses_and_obstacle
            else (obstacle_shape,)
        )

        obstacle_space = spaces.Box(
            low=-1.0, high=1.0, shape=obstacle_shape, dtype=np.float64
        )  # x, y, R

        self.observation_space = spaces.Dict(
            {
                "warehouses": warehouses_space,
                "cameras": self_space,
                "targets": target_space,
                "obstacles": obstacle_space,
            }
        )

        self.obstacles_state = None
        self.warehouses_state = None

    @property
    def description(self):

        if self.warehouse_include_R:
            warehouses_shape_text = "(num_cameras, num_warehouses, 3)"
            if not self.expand_warehouses_and_obstacle:
                warehouses_shape_text = "(num_warehouses, 3)"
        else:
            warehouses_shape_text = "(num_cameras, num_warehouses, 2)"
            if not self.expand_warehouses_and_obstacle:
                warehouses_shape_text = "(num_warehouses, 2)"

        if self.target_include_R and self.target_include_loaded:
            target_shape_text = "(num_cameras, num_targets, 5)"
        elif self.target_include_R or self.target_include_loaded:
            target_shape_text = "(num_cameras, num_targets, 4)"
        else:
            target_shape_text = "(num_cameras, num_targets, 4)"

        if self.expand_warehouses_and_obstacle:
            obstacle_shape_text = "(num_cameras, num_obstacles, 3)"
        else:
            obstacle_shape_text = "(num_obstacles, 3)"

        return {
            "warehouses": "The warehouses state of the environment, including the xy-positions"
            + (" and radius" if self.warehouse_include_R else "")
            + " of the warehouses. Shape: "
            + warehouses_shape_text
            + ".",
            "cameras": "The state of the camera itself, including normalized {x, y, R, phi, theta, Rmax, phimax, thetamax}. Shape: (num_cameras, 8).",
            "targets": "The state of the targets, including normalized {x, y, (optionally R, loaded), seen_flag} for each target. Shape: "
            + target_shape_text
            + ".",
            "obstacles": "The state of the obstacles, including normalized x, y, R for each obstacle. Shape: "
            + obstacle_shape_text
            + ".",
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        self.obstacles_state = None
        self.warehouses_state = None

    def get_warehouses(self):
        if self.warehouses_state is None:
            self.warehouses_state = (
                mate.constants.WAREHOUSES / TERRAIN_SIZE
            )  # (ware_house_x, ware_house_y)*4
            warehouse_radius = (
                mate.constants.WAREHOUSE_RADIUS / TERRAIN_SIZE
            )  # warehouse_radious

            self.warehouses_state = np.expand_dims(
                self.warehouses_state, axis=0
            )  # reshape to (1, num_warehouses, 2)
            self.warehouses_state = np.repeat(
                self.warehouses_state, self.num_cameras, axis=0
            )  # reshape to (num_cameras, num_warehouses, 2)

            if self.warehouse_include_R:
                self.warehouses_state = np.concatenate(
                    [
                        self.warehouses_state,
                        np.full(
                            (self.num_cameras, self.env.unwrapped.num_warehouses, 1),
                            warehouse_radius,
                        ),
                    ],
                    axis=2,
                )  # add warehouse_radious to warehouses state
            if not self.expand_warehouses_and_obstacle:
                self.warehouses_state = self.warehouses_state[0]

    def get_cameras(self, obs):
        (
            cameras_state_x,
            cameras_state_y,
            cameras_state_r,
            Rcos,
            Rsin,
            cameras_state_theta,
            cameras_state_Rmax,
            cameras_state_phimax,
            cameras_state_thetamax,
        ) = obs[:, :].T

        cameras_state_x /= TERRAIN_SIZE
        cameras_state_y /= TERRAIN_SIZE
        cameras_state_R = np.sqrt(Rcos**2 + Rsin**2) / TERRAIN_SIZE
        cameras_state_phi = np.arctan2(Rsin, Rcos) / np.pi
        cameras_state_theta = cameras_state_theta / 180
        cameras_state_phimax /= 180
        cameras_state_Rmax /= TERRAIN_SIZE
        cameras_state_thetamax /= 180
        cameras_state = np.stack(
            [
                cameras_state_x,
                cameras_state_y,
                cameras_state_R,
                cameras_state_phi,
                cameras_state_theta,
                cameras_state_Rmax,
                cameras_state_phimax,
                cameras_state_thetamax,
            ],
            axis=1,
        )  # exclude r

        return cameras_state

    def get_targets(self, obs):
        targets_state = obs[:, :]
        dim = len(targets_state[0])
        targets_state[:, np.arange(0, dim, 5)] /= TERRAIN_SIZE  # x normalization
        targets_state[:, np.arange(1, dim, 5)] /= TERRAIN_SIZE  # y normalization
        targets_state[:, np.arange(2, dim, 5)] /= TERRAIN_SIZE  # R normalization

        exclude_indice = []
        if not self.target_include_R:
            exclude_indice += np.arange(2, dim, 5).tolist()  # R
        if not self.target_include_loaded:
            exclude_indice += np.arange(3, dim, 5).tolist()  # loaded

        targets_state = np.delete(targets_state, exclude_indice, axis=1)

        targets_state = targets_state.reshape(
            self.num_cameras, self.env.unwrapped.num_targets, -1
        )  # reshape to (num_cameras, num_targets, target_state_dim_per_target)

        return targets_state

    def get_obstacles(self):
        if self.obstacles_state is None:
            self.obstacles_state = np.zeros(
                self.env.unwrapped.num_obstacles * 3, dtype=np.float64
            )

            for i in range(self.env.unwrapped.num_obstacles):
                assert isinstance(self.env.unwrapped.obstacles[i], Obstacle)
                self.obstacles_state[i * 3 : (i + 1) * 3] = self.env.unwrapped.obstacles[
                    i
                ].state()

            self.obstacles_state /= TERRAIN_SIZE

            self.obstacles_state = self.obstacles_state.reshape(
                self.env.unwrapped.num_obstacles, 3
            )  # reshape to (num_obstacles, 3)

            if self.expand_warehouses_and_obstacle:
                self.obstacles_state = np.repeat(
                    self.obstacles_state[np.newaxis, :, :], self.num_cameras, axis=0
                )  # reshape to (num_cameras, num_obstacles, 3)

    def observation(self, obs: np.ndarray) -> RefinedObs:
        # PRESERVE OBSERVATION NORMALIZATION
        obs_ = obs.copy()
        ignore_dim = 4  # Nc, Nt, No, self_state_index
        warehouses_dim = self.env.unwrapped.num_warehouses * 2 + 1
        self.get_warehouses()
        ignore_dim += warehouses_dim  # increase ignore_dim for camera state

        # CAMERA STATE OBSERVATION NORMALIZATION
        self_state_dim = 9  # x, y, r, Rcos, Rsin, theta, Rmax, phimax, thetamax
        cameras_state = self.get_cameras(
            obs_[:, ignore_dim : ignore_dim + self_state_dim]
        )
        ignore_dim += self_state_dim  # increase ignore_dim for target state

        # TARGET STATE OBSERVATION NORMALIZATION
        targets_state_dim = 5 * self.env.unwrapped.num_targets  # x, y, R, loaded, flag
        targets_state = self.get_targets(
            obs_[:, ignore_dim : ignore_dim + targets_state_dim]
        )
        ignore_dim += targets_state_dim  # increase ignore_dim for obstacle state

        obstacles_state_dim = self.env.unwrapped.num_obstacles * 4  # x, y, R, flag
        self.get_obstacles()
        ignore_dim += obstacles_state_dim  # x, y, R, flag

        return RefinedObs(
            warehouses=self.warehouses_state,
            cameras=cameras_state,
            targets=targets_state,
            obstacles=self.obstacles_state,
        )


class MateCameraDictObsWrapper(ObservationWrapper):
    def __init__(self, env, warehouse_include_R=False, target_include_R=False, target_include_loaded=False, expand_warehouses_and_obstacle=False):
        super().__init__(env)

        self.refined_observator = MateCameraDictObs(
            env,
            warehouse_include_R=warehouse_include_R,
            target_include_R=target_include_R,
            target_include_loaded=target_include_loaded,
            expand_warehouses_and_obstacle=expand_warehouses_and_obstacle,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.refined_observator.reset(**kwargs)
        refined_obs = self.refined_observator.observation(obs)
        return refined_obs, info

    def observation(self, obs):
        return self.refined_observator.observation(obs)


def combine_targets(targets_list: np.ndarray):
    # targets_list: (num_cameras, num_targets, target_state_dim_per_target)

    targets = targets_list[0]   # T x target_dim

    n_targets = targets.shape[0]

    for camera_targets in targets_list[1:]:
        for i in range(n_targets):
            target_state = camera_targets[i]
            if target_state[-1] == 1:  # if seen_flag is 1, update the target state
                targets[i] = target_state

    return targets


class AgentAsActionWrapper(gym.Wrapper):
    env: MultiCamera
    def __init__(self, env: MultiCamera, agents: list[GoalsBaseAgent]):
        super().__init__(env)
        self.agents = agents

        self.refined_observator = MateCameraDictObs(env)

        self.obs = None
        self.refined_obs = None
        self.info = None

    def reset(self, **kwargs):
        print(self.env)
        obs, info = self.env.reset(**kwargs)
        self.info = None

        for i in range(len(self.agents)):
            self.agents[i].reset(obs[i])

        self.obs = obs
        self.refined_obs = self.refined_observator.observation(obs)

        return self.refined_obs, info

    def step(self, goals: list[list[bool]]):
        # actions = np.ndarray([self.unwrapped.num_cameras, 2])

        # # group observe
        for i in range(len(self.agents)):
            self.agents[i].receive_goals(goals[i])
        #     print(self.obs[i])
        #     self.agents[i].observe(self.obs[i], self.info[i] if self.info is not None else None)

        camera_joint_action = mate.group_step(
            self.env.unwrapped, self.agents, self.obs, self.info
        )

        # group communicate
        # for i in range(len(self.agents)):
        #     self.env.send_messages(self.agents[i].send_requests())
        # for i in range(len(self.agents)):
        #     self.agents[i].receive_requests(self.env.receive_messages(agent=self.agents[i]))
        # for i in range(len(self.agents)):
        #     self.env.send_messages(self.agents[i].send_responses())
        # for i in range(len(self.agents)):
        #     self.agents[i].receive_responses(self.env.receive_messages(agent=self.agents[i]))

        # # group act
        # for i in range(len(self.agents)):
        #     actions[i] = self.agents[i].act(self.obs[i], self.info[i] if self.info is not None else None)


        obs, reward, done, truncated, info = self.env.step(camera_joint_action)
        self.obs = obs
        self.refined_obs = self.refined_observator.observation(obs)
        self.info = info
        return self.refined_obs, reward, done, truncated, info

def main():
    from net import Net, torch

    base_env = gym.make('MultiAgentTracking-v0', config = "MATE-4v8-9.yaml")
    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    env = AgentAsActionWrapper(env, GreedyCameraAgent().spawn(env.unwrapped.num_cameras))

    test = Net()

    camera_joint_observation, _ = env.reset()
    goals = [[True] * env.unwrapped.num_targets] * env.unwrapped.num_cameras

    for i in range(1000):
        camera_joint_observation, target_team_reward, done, truncated, info = env.step(goals)

        run = env.render()
        if not run or done:
            env.close()
            break


if __name__ == "__main__":
    main()
