import numpy as np


class Iteration:

    @staticmethod
    def iterative_policy_evaluation(
        model_s: np.ndarray,
        model_r: np.ndarray,
        model_p: np.ndarray,
        pi: np.ndarray,
        theta: float = 0.000001,
        gamma: float = 0.9999,
        V_param: np.ndarray = None,
    ) -> np.ndarray:
        if V_param is None:
            V_param = np.zeros(model_p.shape[0], dtype=float)
        while True:
            delta = 0.0
            for s_index in range(model_p.shape[0]):
                v = V_param[model_s[s_index]]
                total = 0.0
                for a_index in range(model_p.shape[1]):
                    action_total = 0.0
                    for s_p_index in range(model_p.shape[2]):
                        for r_index in range(model_p.shape[3]):
                            action_total += model_p[
                                s_index, a_index, s_p_index, r_index
                            ] * (model_r[r_index] + gamma * V_param[s_p_index])
                    total += pi[s_index, a_index] * action_total
                V_param[model_s[s_index]] = total
                delta = np.maximum(delta, np.abs(total - v))
            if delta < theta:
                break
        return V_param

    @staticmethod
    def policy_iteration(
        model_s: np.ndarray,
        model_r: np.ndarray,
        model_a: np.ndarray,
        model_p: np.ndarray,
        gamma: float = 0.999,
        theta: float = 0.00001,
    ):
        V = np.zeros(model_p.shape[0], dtype=float)
        pi = np.zeros((len(model_s), len(model_a)))

        for s_index in range(len(model_s)):
            rdm_a_index = np.random.randint(0, len(model_a))
            pi[s_index, rdm_a_index] = 1.0
        while True:
            V = Iteration.iterative_policy_evaluation(
                model_s, model_r, model_p, pi, theta=theta, gamma=gamma, V_param=V
            )
            policy_stable = True
            for s_index in range(len(model_s)):
                old_a_index = np.argmax(pi[s_index])

                best_a_index = None
                best_a_score = 0.0

                for a_index in range(model_p.shape[1]):
                    total = 0.0
                    for s_p_index in range(model_p.shape[2]):
                        for r_index in range(model_p.shape[3]):
                            total += model_p[s_index, a_index, s_p_index, r_index] * (
                                model_r[r_index] + gamma * V[s_p_index]
                            )
                    if best_a_index is None or total > best_a_score:
                        best_a_index = a_index
                        best_a_score = total

                pi[s_index, :] = 0.0
                pi[s_index, best_a_index] = 1.0
                if old_a_index != best_a_index:
                    policy_stable = False

            if policy_stable:
                break

        return pi, V
