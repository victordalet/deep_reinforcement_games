import numpy as np
import time

from src.algo.iteration import Iteration
from src.algo.monte_carlo import MonteCarlo
from src.model.grid_search import GridWorldEnv

A = np.array([0, 1, 2, 3])  # left, right, top, down

S = np.array([[x for x in range(5)] for _ in range(5)])  # agent position

R = np.array([0, -1, 1])  # reward

# Rule :  if agent position = (0,4), reward = -1 ; ele if agent_position = (4,4) ; reward = 1 ; else reward = 0

p = np.zeros((len(S) * len(S[0]), len(A), len(S) * len(S[0]), len(R)))  # probability

c = len(S[0])
for i in range(0, len(S)):
    for j in range(0, len(S[0])):
        if j != 0:
            p[(i * c) + j, 0, (i * c) + j - 1, 0] = 1
        if j != len(S[0]) - 1:
            p[
                (i * c) + j,
                1,
                ((i * c) + j + 1),
                (
                    0
                    if ((i * c) + j + 1) not in (4, 24)
                    else 1 if ((i * c) + j + 1) == 4 else 2
                ),
            ] = 1
        if i != 0:
            p[
                (i * c) + j,
                2,
                (((i - 1) * c) + j),
                0 if (((i - 1) * c) + j) != 4 else 1,
            ] = 1
        if i != len(S) - 1:
            p[
                (i * c) + j,
                3,
                (((i + 1) * c) + j),
                0 if (((i + 1) * c) + j) != 24 else 2,
            ] = 1

print('> export PYTHONPATH="${PYTHONPATH}:src"')

print("TEST GRID SEARCH | ITERATION")

start_time = time.time()

S_format = np.array([s for s in range(len(S) * len(S[0]))])
pi_rsult, V_result = Iteration.policy_iteration(S_format, R, A, p)
end_time = time.time()
print(pi_rsult)
print([float(round(x, 2)) for x in V_result])
for i in range(len(S)):
    for j in range(len(S[0])):
        print(
            (
                "<"
                if pi_rsult[(i * c) + j, 0] == 1.0
                else (
                    ">"
                    if pi_rsult[(i * c) + j, 1] == 1.0
                    else "^" if pi_rsult[(i * c) + j, 2] == 1.0 else "v"
                )
            ),
            end="",
        )
    print("")

print(f"Resolved in :{end_time - start_time}")

print("TEST GRID SEARCH | MONTE CARLO")
game = GridWorldEnv(rows=5, cols=5)
start_time = time.time()
result = MonteCarlo.on_policy_monte_carlo_prediction(game)
end_time = time.time()
print(result)
print(f"Resolved in :{end_time - start_time}")
