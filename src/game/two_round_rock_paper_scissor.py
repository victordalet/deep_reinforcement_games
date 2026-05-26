import numpy as np
import pygame
import os
from src.model.two_round_rock_paper_scissor import TwoRockPaperScissorEnv
from src.game.color import *


class TwoRoundRockPaperScissorGame:
    def __init__(self):
        self.env = TwoRockPaperScissorEnv()

        self.pi = None
        self.Q = None
        self.iteration = 0
        self.total_iterations = 100_000
        self.trajectory = []
        self.current_step = 0

        self.episode_actions = []
        self.episode_rewards = []
        self.episode_opponent_actions = []

        pygame.init()
        self.width = 700
        self.height = 700
        self.fps = 1
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.action_names = ["Rock", "Paper", "Scissor"]

        # Load assets
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        self.asset_images = self._load_assets()

    def _load_assets(self):
        """Load rock, paper, scissor images."""
        images = {}
        asset_files = {0: "rock.png", 1: "paper.png", 2: "scissor.png"}

        for action, filename in asset_files.items():
            filepath = os.path.join(self.assets_dir, filename)
            try:
                img = pygame.image.load(filepath)
                img = pygame.transform.scale(img, (80, 80))  # Resize to 80x80
                images[action] = img
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                images[action] = None

        return images

    def train(self):
        self.env.reset()

        trajectory_state = []
        trajectory_actions = []
        trajectory_rewards = []
        opponent_actions = []

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
            opponent_actions.append(self.env.adversary_last_action)

        # Initialize Q, pi on first episode
        if self.pi is None:
            self.pi = np.ones((2, 3)) / 3
            self.Q = np.zeros((2, 3))
            self.Q_counts = np.zeros((2, 3))

        # Monte Carlo update
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
                self.pi[s, :] = epsilon / 3
                self.pi[s, best_action] = 1 - epsilon + epsilon / 3

            t -= 1

        self.trajectory = trajectory_state
        self.episode_actions = trajectory_actions
        self.episode_rewards = trajectory_rewards
        self.episode_opponent_actions = opponent_actions
        self.current_step = 0

    def draw_round_card(self, x, y, round_num, agent_action, opponent_action, reward):
        card_width = 280
        card_height = 160

        pygame.draw.rect(self.screen, CELL_COLOR, (x, y, card_width, card_height))

        if opponent_action is not None and self.asset_images[opponent_action]:
            self.screen.blit(self.asset_images[opponent_action], (x + 160, y + 65))

    def draw_episode_summary(self):
        summary_y = 350

        # Round 1
        round1_action = (
            self.episode_actions[0] if len(self.episode_actions) > 0 else None
        )
        round1_opponent = (
            self.episode_opponent_actions[0]
            if len(self.episode_opponent_actions) > 0
            else None
        )
        round1_reward = self.episode_rewards[0] if len(self.episode_rewards) > 0 else 0
        self.draw_round_card(
            50, summary_y + 50, 0, round1_action, round1_opponent, round1_reward
        )

        # Round 2
        round2_action = (
            self.episode_actions[1] if len(self.episode_actions) > 1 else None
        )
        round2_opponent = (
            self.episode_opponent_actions[1]
            if len(self.episode_opponent_actions) > 1
            else None
        )
        round2_reward = self.episode_rewards[1] if len(self.episode_rewards) > 1 else 0
        self.draw_round_card(
            380, summary_y + 50, 1, round2_action, round2_opponent, round2_reward
        )

    def draw(self):
        self.screen.fill(BLACK_COLOR)
        self.draw_episode_summary()
        pygame.display.flip()

    def run(self):

        while True:

            if self.iteration > self.total_iterations:
                break

            self.train()
            self.iteration += 1

            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()


if __name__ == "__main__":
    game = TwoRoundRockPaperScissorGame()
    game.run()
