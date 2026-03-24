from mate.agents.base import CameraAgentBase


class GoalsBaseAgent(CameraAgentBase):
    def __init__(
        self,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.goals = None

    def receive_goals(self, goals: list[bool]):
        self.goals = goals

    def reset(self, observation):
        super().reset(observation)

        self.goals = [True] * self.num_opponents


