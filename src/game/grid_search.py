import numpy as np
import pygame
from src.model.grid_search import GridWorldEnv
from src.game.color import *


class GridSearchGame:
    def __init__(self, rows: int = 5, cols: int = 5):
        self.env = GridWorldEnv(rows=rows, cols=cols)
        self.rows = rows
        self.cols = cols

        self.pi = None
        self.Q = None
        self.iteration = 0
        self.total_iterations = 10_000
        self.trajectory = []
        self.current_step = 0

        pygame.init()
        self.width = 500
        self.height = 500
        self.fps = 40
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def train(self):
        self.env.reset()

        trajectory_state = []
        trajectory_actions = []
        trajectory_rewards = []

        while not self.env.is_game_over():
            s = self.env.current_state()
            available = self.env.available_actions()

            if self.pi is not None:
                a_probs = self.pi[s, available]
                a_probs = a_probs / a_probs.sum()
                a = np.random.choice(available, p=a_probs)
            else:
                a = np.random.choice(available)

            prev_score = self.env.score()
            self.env.step(a)
            r = self.env.score() - prev_score

            trajectory_state.append(s)
            trajectory_actions.append(a)
            trajectory_rewards.append(r)

        if self.pi is None:
            self.pi = np.ones((self.rows * self.cols, 4)) / 4
            self.Q = np.zeros((self.rows * self.cols, 4))
            self.Q_counts = np.zeros((self.rows * self.cols, 4))

        gamma = 0.999999
        epsilon = 0.1

        G = 0
        t = len(trajectory_state) - 1
        for s, a, r in zip(
            reversed(trajectory_state),
            reversed(trajectory_actions),
            reversed(trajectory_rewards),
        ):
            G = gamma * G + r
            if (s, a) not in zip(trajectory_state[:t], trajectory_actions[:t]):
                self.Q[s, a] = (self.Q[s, a] * self.Q_counts[s, a] + G) / (
                    self.Q_counts[s, a] + 1
                )
                self.Q_counts[s, a] += 1

                best_action = np.argmax(self.Q[s])
                self.pi[s, :] = epsilon / 4
                self.pi[s, best_action] = 1 - epsilon + epsilon / 4

            t -= 1

        self.trajectory = trajectory_state
        self.current_step = 0

    def draw_grid_world(self):
        cell_size = 80
        start_x = 50
        start_y = 50

        for r in range(self.rows):
            for c in range(self.cols):
                x = start_x + c * cell_size
                y = start_y + r * cell_size

                if (r, c) == (0, self.cols - 1):
                    color = BAD_COLOR
                elif (r, c) == (self.rows - 1, self.cols - 1):
                    color = GOOD_COLOR
                else:
                    color = CELL_COLOR

                pygame.draw.rect(
                    self.screen, color, (x, y, cell_size - 5, cell_size - 5)
                )

        if self.trajectory and self.current_step < len(self.trajectory):
            state = self.trajectory[self.current_step]
            row = state // self.cols
            col = state % self.cols
            x = start_x + col * cell_size + cell_size // 2
            y = start_y + row * cell_size + cell_size // 2
            pygame.draw.circle(self.screen, PLAYER_COLOR, (int(x), int(y)), 15, 3)

    def draw(self):
        self.screen.fill(BLACK_COLOR)
        self.draw_grid_world()
        pygame.display.flip()

    def run(self):

        while True:

            self.train()
            self.iteration += 1

            if self.iteration >= self.total_iterations:
                break

            for _ in range(len(self.trajectory)):
                self.draw()
                pygame.time.delay(1000 // self.fps)
                self.current_step += 1

            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()


if __name__ == "__main__":
    game = GridSearchGame(rows=5, cols=5)
    game.run()
