import numpy as np
from src.model.model_free_env import ModelFreeEnv


class MonteCarlo:

    @staticmethod
    def first_visite_monte_carlo_prediction(
        env: ModelFreeEnv,
        pi: np.ndarray,
        iteration_count: int = 10_000,
        gamma: float = 0.9999,
    ) -> np.ndarray:
        V = np.random.random(env.maximum_states_count())
        returns = [[] for s in range(env.maximum_states_count())]

        for it in range(iteration_count):
            env.reset()

            trajectory_state = []
            trajectory_actions = []
            trajectory_rewards = []

            while not env.is_game_over():
                s = env.current_state()
                available = env.available_actions()
                a_probs = pi[
                    s, available
                ]  # Get probabilities only for available actions
                a_probs = a_probs / a_probs.sum()  # Normalize
                a = np.random.choice(available, p=a_probs)
                prev_score = env.score()
                env.step(a)
                r = env.score() - prev_score

                trajectory_state.append(s)
                trajectory_actions.append(a)
                trajectory_rewards.append(r)

            V[env.current_state()] = 0

            G = 0
            t = len(trajectory_state)
            for s, a, r in zip(
                reversed(trajectory_state),
                reversed(trajectory_actions),
                reversed(trajectory_rewards),
            ):
                t -= 1
                G = gamma * G + r
                if s not in trajectory_state[:t]:
                    returns[s].append(G)
                    V[s] = np.mean(returns[s])

        return V

    @staticmethod
    def on_policy_monte_carlo_prediction(
        env: ModelFreeEnv,
        iteration_count: int = 10_000,
        gamma: float = 0.999999,
        epsilon: float = 0.5,
    ):
        Q = np.random.random((env.maximum_states_count(), env.maximum_actions_count()))
        Q_counts = np.zeros((env.maximum_states_count(), env.maximum_actions_count()))

        pi = (
            np.ones((env.maximum_states_count(), env.maximum_actions_count()))
            / env.maximum_actions_count()
        )

        for it in range(iteration_count):
            env.reset()

            trajectory_state = []
            trajectory_actions = []
            trajectory_rewards = []

            while not env.is_game_over():
                s = env.current_state()
                available = env.available_actions()
                a_probs = pi[
                    s, available
                ]  # Get probabilities only for available actions
                a_probs = a_probs / a_probs.sum()  # Normalize
                a = np.random.choice(available, p=a_probs)
                prev_score = env.score()
                env.step(a)
                r = env.score() - prev_score

                trajectory_state.append(s)
                trajectory_actions.append(a)
                trajectory_rewards.append(r)

            G = 0
            t = len(trajectory_state) - 1
            for s, a, r in zip(
                reversed(trajectory_state),
                reversed(trajectory_actions),
                reversed(trajectory_rewards),
            ):
                G = gamma * G + r
                if (s, a) not in zip(trajectory_state[:t], trajectory_actions[:t]):
                    Q[s, a] = (Q[s, a] * Q_counts[s, a] + G) / (Q_counts[s, a] + 1)
                    Q_counts[s, a] += 1

                    best_action = np.argmax(Q[s])
                    pi[s, :] = epsilon / env.maximum_actions_count()
                    pi[s, best_action] = (
                        1 - epsilon + epsilon / env.maximum_actions_count()
                    )

                t -= 1
        return pi, Q
