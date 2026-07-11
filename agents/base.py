import numpy as np

from mate.agents.base import CameraAgentBase


class HeuristicCameraAgentBase(CameraAgentBase):
    ALGORITHM: str
    def __init__(self, need_history: bool = False, seed=None):
        """Initialize the agent.
        This function will be called only once on initialization.
        """
        super().__init__(seed=seed)
        self.need_history = need_history


class HeuristicsCameraAgent(CameraAgentBase):
    """Heuristic Camera Agent Base

    This class is a base class for heuristic camera agents.
    It provides a common interface for heuristic camera agents to implement.
    """

    def __init__(self, heuristics: list[HeuristicCameraAgentBase], seed=None):
        """Initialize the agent.
        This function will be called only once on initialization.
        """
        super().__init__(seed=seed)

        self.heuristics = heuristics

        self.used_heuristic_idx: int = None

    def receive_heuristic_assignments(self, heuristic_assignment: int):
        """Receive the heuristic assignments for each camera.
        This function will be called before act().
        """
        if heuristic_assignment < 0 or heuristic_assignment >= len(self.heuristics):
            raise ValueError(
                f"Heuristic assignment {heuristic_assignment} is out of range for {len(self.heuristics)} heuristics."
            )
        self.used_heuristic_idx = heuristic_assignment

    def reset(self, observation: np.ndarray):
        for heuristic in self.heuristics:
            heuristic.reset(observation)

    def observe(self, observation: np.ndarray, info: dict | None = None) -> None:
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx or self.heuristics[i].need_history:
                self.heuristics[i].observe(observation, info)

    def act(
        self,
        observation: np.ndarray,
        info: dict | None = None,
        deterministic: bool | None = None,
    ) -> int | np.ndarray:
        """Act based on the observation and info.
        This function will be called after observe().
        """
        if self.used_heuristic_idx is None:
            raise ValueError("Heuristic assignment not received before act()")

        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx:
                res = self.heuristics[i].act(observation, info, deterministic)
            elif self.heuristics[i].need_history:
                self.heuristics[i].act(observation, info, deterministic)

        return res

    def send_requests(self):
        """Send requests to other agents.
        This function will be called after act().
        """
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx:
                res = self.heuristics[i].send_requests()
            elif self.heuristics[i].need_history:
                self.heuristics[i].send_requests()

        return res

    def receive_requests(self, messages: tuple):
        """Receive requests from other agents.
        This function will be called after send_requests().
        """
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx or self.heuristics[i].need_history:
                self.heuristics[i].receive_requests(messages)

    def send_responses(self):
        """Send responses to other agents.
        This function will be called after receive_requests().
        """
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx:
                res = self.heuristics[i].send_responses()
            elif self.heuristics[i].need_history:
                self.heuristics[i].send_responses()

        return res

    def receive_responses(self, messages: tuple):
        """Receive responses from other agents.
        This function will be called after send_responses().
        """
        for i in range(len(self.heuristics)):
            if i == self.used_heuristic_idx or self.heuristics[i].need_history:
                self.heuristics[i].receive_responses(messages)
