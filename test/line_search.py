from src.algo.iteration import Iteration
from src.algo.monte_carlo import MonteCarlo
from src.model.line_search import LineWorldEnv
import numpy as np
import time

line_game = LineWorldEnv()

print("TEST LINE SEARCH | ITERATION")
A = np.array([0, 1])  # left, right
S = np.array([0, 1, 2, 3, 4])  # agent position
R = np.array([0, -1, 1])  # reward
p = np.zeros((len(S), len(A), len(S), len(R)))
p[1, 0, 0, 1] = 1
p[1, 1, 2, 0] = 1
p[2, 0, 1, 0] = 1
p[2, 1, 3, 0] = 1
p[3, 0, 2, 0] = 1
p[3, 1, 4, 2] = 1
start_time = time.time()
pi_rsult, V_result = Iteration.policy_iteration(S, R, line_game.available_actions(), p)
end_time = time.time()
print(pi_rsult)
print(f"Resolved in :{end_time - start_time}")

print("TEST LINE SEARCH | MONTE CARLO")
start_time = time.time()
result = MonteCarlo.on_policy_monte_carlo_prediction(line_game, iteration_count=10_000)
end_time = time.time()
print(result)
print(f"Resolved in :{end_time - start_time}")
