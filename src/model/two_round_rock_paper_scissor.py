import random
from typing import List, Tuple
from src.model.model_free_env import ModelFreeEnv


class TwoRockPaperScissorEnv(ModelFreeEnv):

    def __init__(self):
        self.agent_last_action = None
        self.adversary_last_action = None
        self.nb_run = 0

    def reset(self):
        self.agent_last_action = None
        self.nb_run = 0

    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Invalid action")

        if self.is_game_over():
            raise Exception("Game is already over")

        self.nb_run += 1
        self.agent_last_action = action

    def is_game_over(self) -> bool:
        return self.nb_run >= 2

    def current_state(self) -> int:
        return self.nb_run

    def available_actions(self) -> List[int]:
        return [0, 1, 2]  # rock, paper, scissor

    def score(self) -> float:
        if self.nb_run == 0:
            return 0

        if self.nb_run == 1:
            bot_action = random.choice(self.available_actions())
        if self.nb_run == 2:
            bot_action = self.agent_last_action

        self.adversary_last_action = bot_action

        if self.agent_last_action == bot_action:
            return 0
        if self.agent_last_action == 0:
            if bot_action == 1:  # r < p
                return -1
            return 1  # r > s
        if self.agent_last_action == 1:
            if bot_action == 0:
                return 1  # p > r
            return -1  # p < s
        if self.agent_last_action == 2:
            if bot_action == 0:
                return -1  # s < r
            return 1  # s > p
        return 0

    def maximum_states_count(self) -> int:
        return 2

    def maximum_actions_count(self) -> int:
        return 3
