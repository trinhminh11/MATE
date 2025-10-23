# Standard library imports
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, Type, TypeVar

# Third-party imports
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from numpy.typing import NDArray
from tqdm import tqdm

import gym_agent.utils as utils
from gym_agent.core.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    make_proba_distribution,
)
from gym_agent.core.vec_env.dummy_vec_env import DummyVecEnv
from gym_agent.core.vec_env.subproc_vec_env import SubprocVecEnv

from .buffers import BaseBuffer, ReplayBuffer, RolloutBuffer
from .callbacks import Callbacks
from .main import make
from .polices import ActorCriticPolicy, BasePolicy
from .transforms import EnvWithTransform

ObsType = TypeVar("ObsType", NDArray, dict[str, NDArray])
ActType = TypeVar("ActType", NDArray, dict[str, NDArray])


class Clock:
    def __init__(self):
        self._last_tick_time = time.perf_counter()
        self._frame_rate = 0.0
        self._delta_time = 0.0

    def tick(self, framerate=0):
        """
        Updates the clock and optionally delays to maintain a specific framerate.
        Returns the number of milliseconds passed since the last call to tick().
        """
        current_time = time.perf_counter()
        elapsed_seconds = current_time - self._last_tick_time
        self._last_tick_time = current_time

        # Calculate delta time in milliseconds
        self._delta_time = elapsed_seconds * 1000

        if framerate > 0:
            target_frame_time = 1.0 / framerate
            if elapsed_seconds < target_frame_time:
                delay_seconds = target_frame_time - elapsed_seconds
                time.sleep(delay_seconds)
                # Recalculate delta time after sleep
                current_time = time.perf_counter()
                self._delta_time = (current_time - self._last_tick_time) * 1000
                self._last_tick_time = current_time

        # Calculate current FPS
        if elapsed_seconds > 0:
            self._frame_rate = 1.0 / elapsed_seconds
        else:
            self._frame_rate = float("inf")  # Avoid division by zero

        return int(self._delta_time)

    def get_time(self):
        """
        Returns the time in milliseconds that passed since the last call to tick().
        """
        return int(self._delta_time)

    def get_fps(self):
        """
        Returns the current framerate.
        """
        return self._frame_rate


class AgentBase(ABC, Generic[ObsType, ActType]):
    memory: BaseBuffer
    envs: DummyVecEnv | SubprocVecEnv

    def __init__(
        self,
        policy: BasePolicy,
        env_factory_fn: Callable[[], EnvWithTransform] | str,
        env_kwargs: Optional[dict[str, Any]] = None,
        num_envs: int = 1,
        n_steps: int = 5,
        batch_size: int = 64,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
        async_vectorization: bool = True,
        name=None,
        device: str = "auto",
        seed=None,
    ):
        """Initializes the agent.

        Args:
            policy (BasePolicy): The policy to use.
            env_factory_fn (Callable[[], EnvWithTransform] | str): A function or string to create the environment.
            env_kwargs (Optional[dict[str, Any]], optional): Additional arguments for the environment. Defaults to None.
            num_envs (int, optional): The number of environments to create. Defaults to 1.
            n_steps (int, optional): The number of steps to run before updating the policy. Defaults to 5.
            batch_size (int, optional): The batch size to use for training. Defaults to 64.
            supported_action_spaces (Optional[tuple[type[spaces.Space], ...]], optional): The action spaces supported by the agent. Defaults to None.
            async_vectorization (bool, optional): Whether to use asynchronous vectorization. Defaults to True.
            name (str, optional): The name of the agent. Defaults to None.
            device (str, optional): The device to use for training. Defaults to "auto".
            seed (int, optional): The random seed to use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        if name is None:
            name = self.__class__.__name__

        if not isinstance(name, str):
            raise ValueError("name must be a string")

        if not isinstance(policy, BasePolicy):
            raise ValueError(
                "policy must be an instance of gym_agent.core.policies.BasePolicy"
            )

        self.name = name
        env_kwargs = env_kwargs or {}

        env_kwargs["render_mode"] = "rgb_array"  # force rgb_array mode

        self.env_kwargs = env_kwargs

        if type(env_factory_fn) is str:
            env_factory_fn = lambda **kwargs: make(env_factory_fn, **kwargs)  # noqa: E731

        self.env_factory_fn = env_factory_fn

        if async_vectorization:
            self.envs = SubprocVecEnv(
                [lambda: env_factory_fn(**env_kwargs) for _ in range(num_envs)]
            )
        else:
            self.envs = DummyVecEnv(
                [lambda: env_factory_fn(**env_kwargs) for _ in range(num_envs)]
            )

        if not isinstance(self.env_factory_fn(**env_kwargs), EnvWithTransform):
            raise ValueError(
                "env must be an instance of gym_agent.core.transforms.EnvWithTransform"
            )

        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

        if supported_action_spaces is not None:
            if not isinstance(supported_action_spaces, tuple):
                raise ValueError("supported_action_spaces must be a tuple")
            if not isinstance(self.action_space, supported_action_spaces):
                raise ValueError(
                    f"Action space {self.action_space} is not supported. Supported action spaces are: {supported_action_spaces}"
                )

        utils.check_for_nested_spaces(self.observation_space)
        utils.check_for_nested_spaces(self.action_space)

        self.action_dist = make_proba_distribution(self.action_space)

        # If the action space is continuous, we need to learn the standard deviation
        if isinstance(self.action_dist, DiagGaussianDistribution):
            log_std_init = 0
            self.log_std = nn.Parameter(
                torch.ones(self.action_dist.action_dim) * log_std_init,
                requires_grad=True,
            ).to(self.device)

        self.num_envs = self.envs.num_envs

        self.n_steps = n_steps

        self.policy = policy

        self.device = utils.get_device(device)
        self.seed = seed

        self.batch_size = batch_size

        self.memory = None

        self.timesteps = 0
        self.episodes = 0

        self._mean_score_window = 100
        # history of scores each episode
        self.scores: list[float] = []
        # keep track of each env current running score
        self.current_scores = np.zeros(self.num_envs, dtype=np.float32)

        self.save_kwargs: list[str] = []

        self._last_obs: np.ndarray = None
        self._last_episode_starts: np.ndarray = None

        self.start_time = None

        self.to(self.device)

    def set_mean_score_window(self, window: int):
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")

        self._mean_score_window = window

    @property
    def mean_scores(self):
        if len(self.scores) == 0:
            return []

        if len(self.scores) < self._mean_score_window:
            return [np.mean(self.scores[: i + 1]) for i in range(len(self.scores))]

        mean_scores = []
        cumsum = np.cumsum(self.scores, dtype=float)
        for i in range(len(self.scores)):
            if i < self._mean_score_window:
                mean_scores.append(cumsum[i] / (i + 1))
            else:
                mean_scores.append(
                    (cumsum[i] - cumsum[i - self._mean_score_window])
                    / self._mean_score_window
                )

        return mean_scores

    @property
    def info(self):
        return {
            "scores": self.scores,
            "episodes": self.episodes,
            "total_timesteps": self.timesteps,
        }

    def plot_scores(
        self, filename: Optional[str] = None, rolling_window: Optional[int] = None
    ):
        if rolling_window is None:
            rolling_window = self._mean_score_window

        orig_mean_score_window = self._mean_score_window
        self.set_mean_score_window(rolling_window)
        utils.plot_rl_style(self.scores, self._mean_score_window, filename)
        self.set_mean_score_window(orig_mean_score_window)

    def apply(self, fn: Callable[[nn.Module], None]):
        self.policy.apply(fn)

    def to(self, device):
        self.device = device
        if self.memory:
            self.memory.to(device)
        self.policy.to(device)

        return self

    def add_save_kwargs(self, name: str):
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        if not hasattr(self, name):
            raise ValueError(f"{name} is not an attribute of the agent")

        self.save_kwargs.append(name)

    def save_info(self):
        """
        Get the information to be saved.

        """
        ret = {
            "policy": self.policy.save_info(),
            "scores": self.scores,
            "episodes": self.episodes,
            "total_timesteps": self.timesteps,
        }

        for name in self.save_kwargs:
            ret[name] = getattr(self, name)

        return ret

    def save(self, dir, *post_names):
        """Save the agent's information to a file.

        Args:
            dir (str): The directory to save the file.
            *post_names: Additional strings to append to the filename.
            save_key (list[str], optional): The keys to save. If not provided, defaults to ["policy", "total_timesteps", "scores", "mean_scores", "optimizers"] + self.save_kwargs
        """
        name = self.name

        if not os.path.exists(dir):
            os.makedirs(dir)

        for post_name in post_names:
            name += "_" + str(post_name)

        torch.save(self.save_info(), os.path.join(dir, name + ".pth"))

    def load(self, dir, *post_names):
        """Load the agent's information from a file.

        Args:
            dir (str): The directory to load the file from.
            *post_names: Additional strings to append to the filename.
            load_key (list[str], optional): The keys to load. If not provided, defaults to ["policy", "total_timesteps", "scores", "mean_scores", "optimizers"] + self.save_kwargs.
        """

        if len(post_names) > 0:
            name = self.name + "_" + "_".join(map(str, post_names))  # name_post1_post2
            candidates = [os.path.join(dir, name + ".pth")]
        else:
            candidates = [
                os.path.join(dir, self.name + ".pth"),
                os.path.join(dir, self.name + "_best.pth"),
            ]

        for candidate in candidates:
            if os.path.exists(candidate):
                checkpoint = torch.load(candidate, self.device, weights_only=False)
                break
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {dir} with name {self.name} and post names {post_names} with candidates {candidates}"
            )

        self.policy.load_info(checkpoint["policy"])

        self.episodes = checkpoint["episodes"]
        self.timesteps = checkpoint["total_timesteps"]
        self.scores = checkpoint["scores"]

        for name in self.save_kwargs:
            setattr(self, name, checkpoint[name])

    def reset(self):
        r"""
        This method should call at the start of an episode (after :func:``env.reset``)
        """
        ...

    @abstractmethod
    def predict(self, observations: ObsType, deterministic: bool = True) -> ActType:
        """
        Perform an action based on the given observations.

        Parameters:
            observations (ObsType): The input observations which can be either a numpy array or a dictionary
            * ``NDArray`` shape - `[batch, *obs_shape]`
            * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_obs_shape]`
            deterministic (bool, optional): If True, the action is chosen deterministically. Defaults to True.

        Returns:
            ActType: The action to be performed.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_buffer(self, deterministic: bool, callbacks: Type[Callbacks] = None):
        raise NotImplementedError

    def _setup_fit(
        self,
        n_games: int,
        total_timesteps: int,
        callback: Optional[Callbacks] = None,
        reset_timesteps: bool = True,
        progress_bar: Optional[Type[tqdm]] = tqdm,
        tb_log_name: str = "run",
    ) -> tuple[int, int, tqdm | range, Callbacks]:
        self.start_time = time.time_ns()

        if reset_timesteps:
            self.timesteps = 0
            self.episodes = 0
            self.scores = []
            self.current_scores = np.zeros(self.num_envs, dtype=np.float32)
        else:
            # Make sure training timesteps are ahead of the internal counter
            if total_timesteps:
                total_timesteps += self.timesteps
            else:
                n_games += self.episodes

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_timesteps or self._last_obs is None:
            assert self.envs is not None
            self._last_obs = self.envs.reset()[0]
            self._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

        # Create eval callback if needed
        if callback is None:
            callback = Callbacks(self)

        if total_timesteps is not None:
            loop = (
                progress_bar(range(total_timesteps))
                if progress_bar
                else range(total_timesteps)
            )
        else:
            loop = (
                progress_bar(range(n_games))
                if progress_bar
                else range(n_games)
            )

        return total_timesteps, n_games, loop, callback

    def fit(
        self,
        /,
        n_games: int = None,
        total_timesteps: int = None,
        deterministic=False,
        reset_timesteps: bool = True,
        save_best=False,
        save_every=False,
        save_last=True,
        save_dir="./checkpoints",
        progress_bar: Optional[Type[tqdm]] = tqdm,
        callbacks: Type[Callbacks] = None,
    ):
        if n_games is None and total_timesteps is None:
            raise ValueError("n_games or total_timesteps must be provide")

        total_timesteps, n_games, loop, callbacks = self._setup_fit(
            n_games,
            total_timesteps,
            callbacks,
            reset_timesteps=reset_timesteps,
            progress_bar=progress_bar,
        )

        if progress_bar:
            if total_timesteps:
                loop.update(self.timesteps)
            else:
                loop.update(self.episodes)

        callbacks.on_train_begin()
        self.policy.train()

        best_score = float("-inf")

        counter = 0
        while True:
            counter += 1
            self.reset()
            timesteps, episodes = self.collect_buffer(deterministic, callbacks)

            self.timesteps += timesteps
            self.episodes += episodes

            if len(self.scores) == 0:
                avg_score = None
            else:
                avg_score = np.mean(self.scores[-self._mean_score_window :])

            if save_best and avg_score is not None:
                if avg_score > best_score or (avg_score == best_score and episodes > 0):
                    # only save when improved and at least one env finished an episode
                    # or the best score is updated
                    # to avoid saving too many times when the score is not improved but one env finished an episode
                    # e.g. when the agent is not learning at all
                    # and the score is always 0, but one env finishes an episode every now and then
                    # we don't want to save the model every time that happens
                    # so we only save when the best score is updated
                    best_score = avg_score
                    self.save(save_dir, "best")

            if save_every:
                if counter % save_every == 0:
                    self.save(save_dir, str(counter))

            if progress_bar:
                loop.update(timesteps if total_timesteps else episodes)

                loop.set_postfix(
                    {
                        "episodes": self.episodes,
                        "timesteps": self.timesteps,
                        "avg_score": avg_score,
                        "score": self.scores[-1] if len(self.scores) > 0 else None,
                    }
                )

            if total_timesteps and self.timesteps >= total_timesteps:
                break

            if n_games and self.episodes >= n_games:
                break

        if save_last:
            self.save(save_dir)

        callbacks.on_train_end()

    def play(
        self,
        # env: EnvWithTransform = None,
        env_kwargs: dict[str, Any] = None,
        max_episode_steps: int = None,
        FPS: int = 30,
        stop_if_truncated: bool = True,
        deterministic=True,
        seed=None,
        options: Optional[dict[str, Any]] = None,
        jupyter: bool = False,
    ):
        if jupyter:
            from IPython.display import (
                display,  # pyright: ignore[reportMissingModuleSource] # type
            )
            from PIL import Image
        else:
            import pygame

            pygame.init()

        # _env_kwargs = self.env_kwargs | {"render_mode": "rgb_array" if jupyter else "human", "max_episode_steps": max_episode_steps}

        # env_kwargs = _env_kwargs if env_kwargs is None else env_kwargs  # complete remake, not override
        env_kwargs = (
            self.env_kwargs if env_kwargs is None else self.env_kwargs | env_kwargs
        )  # override
        env_kwargs |= {
            "render_mode": "rgb_array" if jupyter else "human",
            "max_episode_steps": None,
        }  # ensure these two keys

        # if env is None:
        env = self.env_factory_fn(**env_kwargs)

        score = 0
        obs = env.reset(seed=seed, options=options)[0]
        self.reset()

        done = False
        clock = Clock()

        time_step = 0

        with torch.no_grad():
            self.policy.eval()
            while not done:
                time_step += 1

                # use here to ensure max_episode_steps is always applied instead of env internal one
                if max_episode_steps and time_step > max_episode_steps:
                    break

                clock.tick(FPS)
                pixel = env.render()

                if jupyter:
                    display(Image.fromarray(pixel), clear=True)

                if isinstance(env.observation_space, gym.spaces.Dict):
                    _obs = {key: np.expand_dims(obs[key], 0) for key in obs}
                else:
                    _obs = np.expand_dims(obs, 0)

                action = self.predict(_obs, deterministic)

                next_obs, reward, terminated, truncated, info = env.step(action[0])

                done = terminated or (truncated and stop_if_truncated)

                obs = next_obs

                score += reward

                if not jupyter:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True

        if not jupyter:
            pygame.quit()

        return score, info

    def play_jupyter(
        self,
        # env: EnvWithTransform = None,
        env_kwargs: dict[str, Any] = None,
        max_episode_steps: int = None,
        FPS: int = 30,
        stop_if_truncated: bool = True,
        deterministic=True,
        seed=None,
        options: Optional[dict[str, Any]] = None,
    ):
        return self.play(
            # env
            env_kwargs,
            max_episode_steps,
            FPS,
            stop_if_truncated,
            deterministic,
            seed,
            options,
            jupyter=True,
        )

    def distribution(self, action_logits: torch.Tensor) -> Distribution:
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(action_logits, self.log_std)
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            # Here action_logits are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=action_logits)
        else:
            raise ValueError("Invalid action distribution")


class OffPolicyAgent(AgentBase[ObsType, ActType]):
    memory: ReplayBuffer

    def __init__(
        self,
        policy: BasePolicy,
        env_factory_fn: Callable[[], EnvWithTransform] | str,
        env_kwargs: Optional[dict[str, Any]] = None,
        num_envs: int = 1,
        n_steps: int = 1,
        gamma=0.99,
        buffer_size=int(1e5),
        batch_size: int = 64,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
        async_vectorization: bool = True,
        name=None,
        device="auto",
        seed=None,
    ):
        super().__init__(
            policy,
            env_factory_fn=env_factory_fn,
            env_kwargs=env_kwargs,
            num_envs=num_envs,
            n_steps=n_steps,
            batch_size=batch_size,
            supported_action_spaces=supported_action_spaces,
            async_vectorization=async_vectorization,
            name=name,
            device=device,
            seed=seed,
        )

        self.gamma = gamma

        self.memory = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            num_envs=self.num_envs,
            seed=self.seed,
        )

    @abstractmethod
    def learn(
        self,
    ) -> None:
        """
        Abstract method to be implemented by subclasses for learning from a batch of experiences.
        use self.memory.sample(self.batch_size) to get a batch of experiences.
        """
        ...

    def collect_buffer(self, deterministic: bool, callbacks: Type[Callbacks]):
        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_begin()

        time_steps = 0
        episodes = 0

        self.policy.train()  # ensure in train mode
        for _ in range(self.n_steps):
            time_steps += 1

            with torch.no_grad():
                actions = self.predict(self._last_obs, deterministic)
            next_obs, rewards, terminated, truncated, info = self.envs.step(actions)

            self.memory.add(self._last_obs, actions, rewards, terminated)

            self._last_obs = next_obs
            self._last_episode_starts = np.array(
                terminated | truncated
            )  # episode starts is just done

            self.current_scores += np.array(rewards, dtype=np.float32)

            # if an env is done, record the score and reset the current score for that env
            self.scores.extend(self.current_scores[self._last_episode_starts].tolist())
            # reset the current score for that env
            self.current_scores[self._last_episode_starts] = 0.0
            # count how many episodes finished
            episodes += np.sum(self._last_episode_starts)

        if len(self.memory) >= self.batch_size: # if collect enough data, start learning 
            callbacks.on_learn_begin()
            self.learn()
            callbacks.on_learn_end()

        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_end()

        return time_steps, episodes


class OnPolicyAgent(AgentBase[ObsType, ActType]):
    """
    This is a base class for on-policy agents.
    Because there are on-policy agents that are not actor-critic, so this is a placeholder for the future on-policy implementations if they arise.
    """

    pass


class ActorCriticPolicyAgent(AgentBase[ObsType, ActType]):
    """This is a base class for actor-critic agents.
    You can consider this class as an on-policy agent with a value function.
    """

    memory: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: ActorCriticPolicy,
        env_factory_fn: Callable[[], EnvWithTransform] | str,
        env_kwargs: Optional[dict[str, Any]] = None,
        n_steps: int = 5,
        num_envs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
        async_vectorization: bool = True,
        name=None,
        device="auto",
        seed=None,
    ):
        super().__init__(
            policy,
            env_factory_fn=env_factory_fn,
            env_kwargs=env_kwargs,
            num_envs=num_envs,
            n_steps=n_steps,
            batch_size=n_steps * num_envs,
            supported_action_spaces=supported_action_spaces,
            async_vectorization=async_vectorization,
            name=name,
            device=device,
            seed=seed,
        )

        self.memory = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            num_envs=self.num_envs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            seed=self.seed,
        )

        self._last_obs = None

    @abstractmethod
    def predict(self, state: ObsType, deterministic: bool = True) -> ActType:
        pass

    @abstractmethod
    def learn(self) -> None:
        """
        Perform learning using the experiences stored in the memory buffer.

        This method should be overridden by subclasses to implement specific learning algorithms.
        The method should utilize the experiences stored in `self.memory` to update the agent's policy.
        use self.memory.get(batch_size) to get a generator that yields batches of experiences.
        """
        raise NotImplementedError

    def evaluate_actions(
        self, obs: ObsType, actions: ActType
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions using the current policy.

        Args:
            obs (ObsType): The input observations which can be either a numpy array or a dictionary
                * ``NDArray`` shape - `[batch, *obs_shape]`
                * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_obs_shape]`
            actions (ActType): The actions to evaluate which can be either a numpy array or a dictionary
                * ``NDArray`` shape - `[batch, *action_shape]`
                * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_action_shape]`
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - action_log_probs (torch.Tensor): The log probabilities of the actions.
                - values (torch.Tensor): The value estimates for the observations.
                - entropy (torch.Tensor): The entropy of the action distribution.
        """
        action_logits, value_logits = self.policy.forward(obs)

        distribution = self.distribution(action_logits)

        action_log_probs = distribution.log_prob(actions)
        values = value_logits.squeeze(-1)
        entropy = distribution.entropy()

        return action_log_probs, values, entropy

    def collect_buffer(self, deterministic: bool, callbacks: Type[Callbacks] = None):
        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_begin()

        self.memory.reset()

        time_steps = 0
        episodes = 0

        self.policy.train()  # ensure in train mode
        for _ in range(self.n_steps):
            time_steps += 1

            with torch.no_grad():
                action_logits, value_logits = self.policy.forward(
                    utils.to_torch(self._last_obs, self.device)
                )

                distribution = self.distribution(action_logits)

                # get the action
                if deterministic:
                    actions = distribution.mode()
                else:
                    actions = distribution.sample()

                log_probs = distribution.log_prob(actions).cpu().numpy()
                actions = actions.cpu().numpy()
                values = value_logits.squeeze(-1).cpu().numpy()

            next_obs, rewards, terminated, truncated, info = self.envs.step(actions)

            self.memory.add(
                self._last_obs,
                actions,
                rewards,
                values,
                log_probs,
                self._last_episode_starts,
            )

            self._last_obs = next_obs
            self._last_episode_starts = np.array(
                terminated | truncated
            )  # episode starts is just done

            self.current_scores += np.array(rewards, dtype=np.float32)

            # if an env is done, record the score and reset the current score for that env
            self.scores.extend(self.current_scores[self._last_episode_starts].tolist())
            # reset the current score for that env
            self.current_scores[self._last_episode_starts] = 0.0
            # count how many episodes finished
            episodes += np.sum(self._last_episode_starts)

        # compute the value of the last observation
        with torch.no_grad():
            values = (
                self.policy.forward_critic(utils.to_torch(self._last_obs, self.device))
                .squeeze(-1)
                .cpu()
                .numpy()
            )

        self.memory.calc_advantages_and_returns(
            last_values=values, last_terminals=self._last_episode_starts
        )
        callbacks.on_learn_begin()
        self.learn()
        callbacks.on_learn_end()

        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_end()

        return time_steps, episodes
