import mate
from mate.agents import GreedyTargetAgent
from mate.constants import TERRAIN_SIZE
from mate.entities import Obstacle
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium import spaces

import numpy as np

class RefinedObs:
    def __init__(self, preserved, cameras, targets, obstacles, warehouse_include_R=False, target_include_R=False, target_include_loaded=False, expand_preserved_and_obstacle=True):
        self.preserved = preserved
        self.cameras = cameras
        self.targets = targets
        self.obstacles = obstacles

        self.warehouse_include_R = warehouse_include_R
        self.target_include_R = target_include_R
        self.target_include_loaded = target_include_loaded
        self.expand_preserved_and_obstacle = expand_preserved_and_obstacle

    def __getitem__(self, key):
        return getattr(self, key)

    def items(self):
        return {
            "preserved": self.preserved,
            "cameras": self.cameras,
            "targets": self.targets,
            "obstacles": self.obstacles,
        }.items()

    def keys(self):
        return ["preserved", "cameras", "targets", "obstacles"]

    def values(self):
        return [self.preserved, self.cameras, self.targets, self.obstacles]

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        for key in self.keys():
            yield key, getattr(self, key)

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        return key in self.keys()

    @property
    def description(self):

        if self.warehouse_include_R:
            preserved_shape_text = "(num_cameras, num_warehouses, 3)"
            if not self.expand_preserved_and_obstacle:
                preserved_shape_text = "(num_warehouses, 3)"
        else:
            preserved_shape_text = "(num_cameras, num_warehouses, 2)"
            if not self.expand_preserved_and_obstacle:
                preserved_shape_text = "(num_warehouses, 2)"

        if self.target_include_R and self.target_include_loaded:
            target_shape_text = "(num_cameras, num_targets, 5)"
        elif self.target_include_R or self.target_include_loaded:
            target_shape_text = "(num_cameras, num_targets, 4)"
        else:
            target_shape_text = "(num_cameras, num_targets, 4)"

        if self.expand_preserved_and_obstacle:
            obstacle_shape_text = "(num_cameras, num_obstacles, 3)"
        else:
            obstacle_shape_text = "(num_obstacles, 3)"


        return {
            "preserved": "The preserved state of the environment, including the xy-positions" + (" and radius" if self.warehouse_include_R else "") + " of the warehouses. Shape: " + preserved_shape_text + ".",
            "cameras": "The state of the camera itself, including normalized {x, y, R, phi, theta, Rmax, phimax, thetamax}. Shape: (num_cameras, 8).",
            "targets": "The state of the targets, including normalized {x, y, (optionally R, loaded), seen_flag} for each target. Shape: " + target_shape_text + ".",
            "obstacles": "The state of the obstacles, including normalized x, y, R for each obstacle. Shape: " + obstacle_shape_text + ".",
        }

class MateCameraDictObsWrapper(ObservationWrapper):
    def __init__(self, env, warehouse_include_R=False, target_include_R=False, target_include_loaded=False, expand_preserved_and_obstacle=False):
        """
        env: the mate environment to wrap
        warehouse_include_R: whether to include the radius of the warehouse in the preserved state
        target_include_R: whether to include the radius of the target in the target state
        target_include_loaded: whether to include the loaded flag of the target in the target state
        expand_preserved_and_obstacle: whether to expand the preserved and obstacle state to have a separate entry for each camera, or to keep them shared among all cameras (if False, the preserved and obstacle state will be the same for all cameras)
        """
        super().__init__(env)
        self.target_include_R = target_include_R
        self.target_include_loaded = target_include_loaded
        self.expand_preserved_and_obstacle = expand_preserved_and_obstacle
        self.warehouse_include_R = warehouse_include_R

        self.num_cameras = self.unwrapped.num_cameras

        preserved_shape = self.unwrapped.num_warehouses*2 + (1 if warehouse_include_R else 0)
        preserved_shape = (self.num_cameras, preserved_shape) if expand_preserved_and_obstacle else (preserved_shape,)

        preserved_space = spaces.Box(low=-1.0, high=1.0, shape=preserved_shape, dtype=np.float64)
        self_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_cameras, 8), dtype=np.float64)   # x, y, R, phi, theta, Rmax, phimax, thetamax

        target_dim = 3
        if target_include_R:
            target_dim += 1
        if target_include_loaded:
            target_dim += 1
        target_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_cameras, self.unwrapped.num_targets*target_dim,), dtype=np.float64)

        obstacle_shape = self.unwrapped.num_obstacles*3
        obstacle_shape = (self.num_cameras, obstacle_shape) if expand_preserved_and_obstacle else (obstacle_shape,)

        obstacle_space = spaces.Box(low=-1.0, high=1.0, shape=obstacle_shape, dtype=np.float64)   # x, y, R


        self.observation_space = spaces.Dict({
            "preserved": preserved_space,
            "cameras": self_space,
            "targets": target_space,
            "obstacles": obstacle_space,
        })

        self.obstacles_state = None
        self.preserved_state = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ):
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        self.obstacles_state = None
        self.preserved_state = None
        obs, info = self.env.reset(seed=seed, options=options)

        obs = self.observation(obs)
        return obs, info



    def observation(self, obs) -> RefinedObs:
        # PRESERVE OBSERVATION NORMALIZATION
        ignore_dim = 4 # Nc, Nt, No, self_state_index
        preserved_dim = self.unwrapped.num_warehouses*2+1
        if self.preserved_state is None:
            self.preserved_state = obs[:, ignore_dim:ignore_dim + preserved_dim -1] / TERRAIN_SIZE  # (ware_house_x, ware_house_y)*4
            warehouse_radious = obs[0, ignore_dim + preserved_dim -1] / TERRAIN_SIZE  # warehouse_radious

            self.preserved_state = self.preserved_state.reshape(self.num_cameras, self.unwrapped.num_warehouses, -1)   # reshape to (num_cameras, preserved_dim)

            if self.warehouse_include_R:
                self.preserved_state = np.concatenate([self.preserved_state, np.full((self.num_cameras, self.unwrapped.num_warehouses, 1), warehouse_radious)], axis=2)   # add warehouse_radious to preserved state

            if not self.expand_preserved_and_obstacle:
                self.preserved_state = self.preserved_state[0]

        ignore_dim += preserved_dim     # increase ignore_dim for camera state

        # CAMERA STATE OBSERVATION NORMALIZATION
        self_state_dim = 9    # x, y, r, Rcos, Rsin, theta, Rmax, phimax, thetamax
        self_state_x, self_state_y, self_state_r, Rcos, Rsin, self_state_theta, self_state_Rmax, self_state_phimax, self_state_thetamax = obs[:, ignore_dim:ignore_dim+self_state_dim].T

        self_state_x /= TERRAIN_SIZE
        self_state_y /= TERRAIN_SIZE
        self_state_R = np.sqrt(Rcos**2 + Rsin**2) / TERRAIN_SIZE
        self_state_phi = np.arctan2(Rsin, Rcos) / np.pi
        self_state_theta = self_state_theta / 180
        self_state_phimax /= 180
        self_state_Rmax /= TERRAIN_SIZE
        self_state_thetamax /= 180
        self_state = np.stack([self_state_x, self_state_y, self_state_R, self_state_phi, self_state_theta, self_state_Rmax, self_state_phimax, self_state_thetamax], axis=1)      # exclude r
        ignore_dim += self_state_dim   # increase ignore_dim for target state

        # TARGET STATE OBSERVATION NORMALIZATION
        target_state_dim = 5 * self.unwrapped.num_targets    # x, y, R, loaded, flag
        target_state = obs[:, ignore_dim:ignore_dim+target_state_dim]

        target_state[:, np.arange(0, target_state_dim, 5)] /= TERRAIN_SIZE  # x normalization
        target_state[:, np.arange(1, target_state_dim, 5)] /= TERRAIN_SIZE  # y normalization
        target_state[:, np.arange(2, target_state_dim, 5)] /= TERRAIN_SIZE  # R normalization

        exclude_indice = []
        if not self.target_include_R:
            exclude_indice += np.arange(2, target_state_dim, 5).tolist()    # R
        if not self.target_include_loaded:
            exclude_indice += np.arange(3, target_state_dim, 5).tolist()    # loaded

        target_state = np.delete(target_state, exclude_indice, axis=1)


        target_state = target_state.reshape(self.num_cameras, self.unwrapped.num_targets, -1)   # reshape to (num_cameras, num_targets, target_state_dim_per_target)

        ignore_dim += target_state_dim   # increase ignore_dim for obstacle state

        # OBSTACLE STATE OBSERVATION NORMALIZATION
        if self.obstacles_state is None:
            self.obstacles_state = np.zeros(self.unwrapped.num_obstacles*3, dtype=np.float64)

            for i in range(self.unwrapped.num_obstacles):
                assert isinstance(self.unwrapped.obstacles[i], Obstacle)
                self.obstacles_state[i*3:(i+1)*3] = self.unwrapped.obstacles[i].state()

            self.obstacles_state /= TERRAIN_SIZE

            self.obstacles_state = self.obstacles_state.reshape(self.unwrapped.num_obstacles, 3)   # reshape to (num_obstacles, 3)

            if self.expand_preserved_and_obstacle:
                self.obstacles_state = np.repeat(self.obstacles_state[np.newaxis, :, :], self.num_cameras, axis=0)

        ignore_dim += self.unwrapped.num_obstacles*4   # x, y, R, flag


        obs = RefinedObs(
            preserved=self.preserved_state,
            cameras=self_state,
            targets=target_state,
            obstacles=self.obstacles_state,
            warehouse_include_R=self.warehouse_include_R,
            target_include_R=self.target_include_R,
            target_include_loaded=self.target_include_loaded,
            expand_preserved_and_obstacle=self.expand_preserved_and_obstacle,
        )

        return obs


MAX_EPISODE_STEPS = 4000


def main():
    base_env = gym.make('MultiAgentTracking-v0', config = "MATE-4v8-9.yaml", render_mode='rgb_array')

    env: mate.MultiAgentTracking = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())
    env = MateCameraDictObsWrapper(env, expand_preserved_and_obstacle=False)

    obs, _ = env.reset()

    for key, value in obs.items():
        print(key, value.shape)


    for key, value in obs.description.items():
        print(key, value)
    # for _ in range(100):
    #     env.step(np.zeros([env.unwrapped.num_cameras, 2]))
    #     if _ > 50:
    #         arr = env.render()

if __name__ == '__main__':
    main()


