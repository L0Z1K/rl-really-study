from enum import Enum
import random

import pandas as pd


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


MAZE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]
WIDTH, HEIGHT = len(MAZE[0]), len(MAZE)
START = (5, 3)
END = (0, 8)


def epsilon_greedy(s, Q, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(list(Action))
    else:
        return max(list(Action), key=lambda a: Q[s, a])


# Dyna-Q Algorithm

# 1. Initialize Q(s, a) and Model(s, a)
Q = {}
Model = {}
for x in range(HEIGHT):
    for y in range(WIDTH):
        Q[(x, y)] = {}
        for a in Action:
            Q[(x, y), a] = random.random() / 100

data = []
for N in [0, 5, 50]:
    results = []
    for episode in range(1000):
        steps = 0
        # 2. Do forever:
        s = START
        while True:
            # (b) a <- epsilon-greedy(s, Q)
            a = epsilon_greedy(s, Q)
            # (c) Take action a; observe resultant reward, r, and state, s'
            x, y = s
            if a == Action.UP and x > 0 and MAZE[x - 1][y] == 0:
                x -= 1
            elif a == Action.DOWN and x < HEIGHT - 1 and MAZE[x + 1][y] == 0:
                x += 1
            elif a == Action.LEFT and y > 0 and MAZE[x][y - 1] == 0:
                y -= 1
            elif a == Action.RIGHT and y < WIDTH - 1 and MAZE[x][y + 1] == 0:
                y += 1
            s_prime = (x, y)
            r = -1 if s_prime != END else 0
            # (d) Q(s, a) <- Q(s, a) + alpha[r + gamma * max_a' Q(s', a') - Q(s, a)]
            alpha = 0.1
            gamma = 0.95
            Q[s, a] += alpha * (
                r + gamma * max([Q[s_prime, a_prime] for a_prime in Action]) - Q[s, a]
            )
            # (e) Model(s, a) <- r, s'
            if s not in Model:
                Model[s] = {}
            Model[s][a] = (r, s_prime)
            # (f) Repeat n times:
            for _ in range(N):
                # (i) s <- random previously observed state
                s_ = random.choice(list(Model.keys()))
                # (ii) a <- random action previously taken in s
                a_ = random.choice(list(Model[s_].keys()))
                # (iii) r, s' <- Model(s, a)
                r_, s_prime_ = Model[s_][a_]
                # (iv) Q(s, a) <- Q(s, a) + alpha[r + gamma * max_a' Q(s', a') - Q(s, a)]
                Q[s_, a_] += alpha * (
                    r
                    + gamma * max([Q[s_prime_, a_prime] for a_prime in Action])
                    - Q[s_, a_]
                )

            # (g) until s is terminal
            if s_prime == END:
                break
            # (a) s <- current (nonterminal) state
            s = s_prime
            steps += 1
        results.append(steps)
    data.append(results)

df = pd.DataFrame(data, index=["N=0", "N=5", "N=50"]).T
df.to_csv("result.csv", index=False)
