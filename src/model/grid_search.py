from typing import List, Tuple
from src.model.model_free_env import ModelFreeEnv


class GridWorldEnv(ModelFreeEnv):
    def __init__(self, rows: int = 5, cols: int = 5):
        self.rows = rows
        self.cols = cols
        self.agent_pos: Tuple[int, int] = (0, 0)

    def reset(self):
        self.agent_pos = (0, 0)

    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception("Invalid action")

        if self.is_game_over():
            raise Exception("Game is already over")

        row, col = self.agent_pos

        # 0 = up, 1 = down, 2 = left, 3 = right
        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1

        self.agent_pos = (row, col)

        return self.current_state(), self.score(), self.is_game_over()

    def is_game_over(self) -> bool:
        return self.agent_pos in [(0, self.cols - 1), (self.rows - 1, self.cols - 1)]

    def current_state(self) -> int:
        row, col = self.agent_pos
        return row * self.cols + col

    def available_actions(self) -> List[int]:
        if self.is_game_over():
            return []

        row, col = self.agent_pos
        actions = []

        if row > 0:
            actions.append(0)  # up
        if row < self.rows - 1:
            actions.append(1)  # down
        if col > 0:
            actions.append(2)  # left
        if col < self.cols - 1:
            actions.append(3)  # right

        return actions

    def score(self) -> float:
        if self.agent_pos == (0, self.cols - 1):
            return -1
        elif self.agent_pos == (self.rows - 1, self.cols - 1):
            return 1
        return 0

    def maximum_states_count(self) -> int:
        return self.rows * self.cols

    def maximum_actions_count(self) -> int:
        return 4

    def pretty_print(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    print("X", end="")
                elif (r, c) == (0, self.cols - 1):
                    print("-", end="")
                elif (r, c) == (self.rows - 1, self.cols - 1):
                    print("+", end="")
                else:
                    print("_", end="")
            print()
