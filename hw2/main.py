import random

import numpy as np
import pandas as pd


class Arm:
    """Returns 1 with probability p, 0 otherwise."""

    def __init__(self, p: float):
        self.p = p

    def pull(self):
        return 1 if random.random() < self.p else 0


class Greedy:
    def __init__(self, arms: list[Arm]):
        self.arms = arms
        self.his = [[0, 0] for _ in range(len(self.arms))]  # tot reward, num pulls

    def Q(self, i: int) -> float:
        return self.his[i][0] / self.his[i][1] if self.his[i][1] > 0 else 0

    def action(self) -> int:
        q = [self.Q(i) for i in range(len(self.arms))]
        idxes = [i for i, qq in enumerate(q) if qq == max(q)]
        return random.choice(idxes)

    def update(self, a: int, r: int) -> None:
        self.his[a][0] += r
        self.his[a][1] += 1


class e_Greedy:
    def __init__(self, epsilon: float, arms: list[Arm]):
        self.epsilon = epsilon
        self.arms = arms
        self.his = [[0, 0] for _ in range(len(self.arms))]

    def Q(self, i: int) -> float:
        return self.his[i][0] / self.his[i][1] if self.his[i][1] > 0 else 0

    def action(self) -> int:
        if random.random() < self.epsilon:
            return random.choice(range(len(self.arms)))
        else:
            q = [self.Q(i) for i in range(len(self.arms))]
            idxes = [i for i, qq in enumerate(q) if qq == max(q)]
            return random.choice(idxes)

    def update(self, a: int, r: int) -> None:
        self.his[a][0] += r
        self.his[a][1] += 1


class UCB:
    def __init__(self, c: float, arms: list[Arm]):
        self.c = c
        self.arms = arms
        self.his = [[0, 0] for _ in range(len(self.arms))]
        self.t = 0

    def Q(self, i: int) -> float:
        return self.his[i][0] / self.his[i][1] if self.his[i][1] > 0 else 0

    def U(self, i: int) -> float:
        return (
            (np.log(self.t + 1) / self.his[i][1]) ** 0.5
            if self.his[i][1] > 0
            else np.inf
        )

    def action(self) -> int:
        q = [self.Q(i) + self.c * self.U(i) for i in range(len(self.arms))]
        idxes = [i for i, qq in enumerate(q) if qq == max(q)]
        return random.choice(idxes)

    def update(self, a: int, r: int) -> None:
        self.his[a][0] += r
        self.his[a][1] += 1
        self.t += 1


class ThompsonSampling:
    def __init__(self, arms: list[Arm]):
        self.arms = arms
        self.his = [
            [1, 1] for _ in range(len(self.arms))
        ]  # alpha(win_count + 1), beta(lose_count + 1)

    def Q(self, i: int) -> int:
        return np.random.beta(self.his[i][0], self.his[i][1])

    def action(self) -> int:
        q = [self.Q(i) for i in range(len(self.arms))]
        idxes = [i for i, qq in enumerate(q) if qq == max(q)]
        return random.choice(idxes)

    def update(self, a: int, r: int) -> None:
        if r == 0:
            self.his[a][1] += 1
        else:
            self.his[a][0] += 1


if __name__ == "__main__":
    result = []
    for i in range(1, 10):
        for j in range(1, 10):
            A = Arm(i / 10)
            B = Arm(j / 10)

            methods = {
                "Greedy": Greedy([A, B]),
                "Ɛ-Greedy": e_Greedy(0.1, [A, B]),
                "UCB": UCB(np.sqrt(2), [A, B]),
                "Thompson Sampling": ThompsonSampling([A, B]),
            }

            for name, method in methods.items():
                score = 0
                for trial in range(5):
                    tot_rewards = 0
                    for _ in range(1000):
                        a = method.action()
                        r = method.arms[a].pull()
                        method.update(a, r)
                        tot_rewards += r
                    tot_rewards /= 1000
                    score += tot_rewards
                score /= 5
                result.append([i / 10, j / 10, name, score])

    pd.DataFrame(result, columns=["E[P(A)]", "E[P(B)]", "Method", "score"]).to_csv(
        "result.csv", index=False
    )
