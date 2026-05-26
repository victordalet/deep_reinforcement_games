from typing import List
from src.model.model_free_env import ModelFreeEnv


class LineWorldEnv(ModelFreeEnv):
    def __init__(self, num_cells: int = 5):
        self.num_cells = num_cells
        self.agent_pos = num_cells // 2

    def reset(self):
        self.agent_pos = self.num_cells // 2

    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Invalid action")

        if self.is_game_over():
            raise Exception("Game is already over")

        if action == 0:
            self.agent_pos -= 1
        elif action == 1:
            self.agent_pos += 1

    def is_game_over(self) -> bool:
        return self.agent_pos == self.num_cells - 1 or self.agent_pos == 0

    def current_state(self) -> int:
        return self.agent_pos

    def available_actions(self) -> List[int]:
        return [0, 1]

    def score(self) -> float:
        if self.agent_pos == self.num_cells - 1:
            return 1
        elif self.agent_pos == 0:
            return -1
        return 0

    def maximum_states_count(self) -> int:
        return self.num_cells

    def maximum_actions_count(self) -> int:
        return 2

    def pretty_print(self):
        for s in range(self.num_cells):
            if s == self.agent_pos:
                print("X", end="")
            else:
                print("_", end="")
        print()
