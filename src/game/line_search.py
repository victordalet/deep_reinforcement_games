import numpy as np
import pygame
from src.model.line_search import LineWorldEnv
from src.game.color import *


class LineSearchGame:
    def __init__(self, num_cells: int = 5):
        self.env = LineWorldEnv(num_cells=num_cells)
        self.num_cells = num_cells

        self.pi = None
        self.Q = None
        self.iteration = 0
        self.total_iterations = 10_000
        self.trajectory = []
        self.current_step = 0

        pygame.init()
        self.width = 800
        self.height = 600
        self.fps = 40
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Line World - Monte Carlo Training")
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
            self.pi = np.ones((self.num_cells, 2)) / 2
            self.Q = np.zeros((self.num_cells, 2))
            self.Q_counts = np.zeros((self.num_cells, 2))

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
                self.pi[s, :] = epsilon / 2
                self.pi[s, best_action] = 1 - epsilon + epsilon / 2

            t -= 1

        self.trajectory = trajectory_state
        self.current_step = 0

    def draw_line_world(self):
        cell_w = 100
        start_x = (self.width - self.num_cells * cell_w) // 2
        cell_h = 60
        y_top = self.height // 2

        for i in range(self.num_cells):
            x = start_x + i * cell_w
            rect = pygame.Rect(x, y_top, cell_w - 10, cell_h)

            if i == 0:
                color = BAD_COLOR
            elif i == self.num_cells - 1:
                color = GOOD_COLOR
            else:
                color = CELL_COLOR

            pygame.draw.rect(self.screen, color, rect)

        if self.trajectory and self.current_step < len(self.trajectory):
            agent_pos = self.trajectory[self.current_step]
            if 0 <= agent_pos < self.num_cells:
                rect_x = start_x + agent_pos * cell_w
                rect_w = cell_w - 10
                rect = pygame.Rect(rect_x, y_top, rect_w, cell_h)
                agent_x = rect.x + rect.width // 2
                agent_y = rect.y + rect.height // 2
                pygame.draw.circle(
                    self.screen, PLAYER_COLOR, (int(agent_x), int(agent_y)), 12
                )

    def draw(self):
        self.screen.fill(BLACK_COLOR)

        self.draw_line_world()

        pygame.display.flip()

    def run(self):

        while True:

            self.train()
            self.iteration += 1

            for _ in range(len(self.trajectory)):
                self.draw()
                pygame.time.delay(1000 // self.fps)
                self.current_step += 1

            if self.iteration >= self.total_iterations:
                break

            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()


if __name__ == "__main__":
    game = LineSearchGame(num_cells=5)
    game.run()
