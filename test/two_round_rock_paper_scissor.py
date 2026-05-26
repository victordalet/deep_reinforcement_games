from src.model.two_round_rock_paper_scissor import TwoRockPaperScissorEnv
from src.algo.monte_carlo import MonteCarlo
import numpy as np
import time

print("TWO ROUND ROCK PAPER SCISSOR | MONTE CARLO")
game = TwoRockPaperScissorEnv()
start_time = time.time()
result = MonteCarlo.on_policy_monte_carlo_prediction(game, iteration_count=100_000)
end_time = time.time()
print(result)
action = ["rock", "paper", "scissor"]
print(
    f"best action round 1 {action[np.argmax(result[1][0])]}, round 2 {action[np.argmax(result[1][1])]}"
)
print(f"Resolved in {end_time - start_time} ")
