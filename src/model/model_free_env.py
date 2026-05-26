from typing import List


class ModelFreeEnv:

    def reset(self):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    def is_game_over(self) -> bool:
        raise NotImplementedError

    def current_state(self) -> int:
        raise NotImplementedError

    def available_actions(self) -> List[int]:
        raise NotImplementedError

    def score(self) -> float:
        raise NotImplementedError

    def maximum_states_count(self) -> int:
        raise NotImplementedError

    def maximum_actions_count(self) -> int:
        raise NotImplementedError
