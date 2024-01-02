import numpy as np


class GridWorld:
    def __init__(self):
        self.grid_size = 4
        self.states = [
            (i, j)
            for i in range(1, self.grid_size + 1)
            for j in range(1, self.grid_size + 1)
        ]
        self.terminal_states = [(4, 4)]

    def is_terminal(self, state):
        return state in self.terminal_states

    def step(self, state, action):
        if self.is_terminal(state):
            return state, 0

        next_state = list(state)
        if action == "up" and state[0] > 1:
            next_state[0] -= 1
        elif action == "down" and state[0] < self.grid_size:
            next_state[0] += 1
        elif action == "left" and state[1] > 1:
            next_state[1] -= 1
        elif action == "right" and state[1] < self.grid_size:
            next_state[1] += 1

        next_state = tuple(next_state)
        reward = 10 if next_state == (4, 4) else -1
        return next_state, reward


def behavior_policy(state):
    actions = ["up", "down", "left", "right"]
    return np.random.choice(actions)


def target_policy(state):
    actions = []
    if state[1] < 4:
        actions.append("right")
    if state[0] > 1:
        actions.append("up")
    actions.extend(["left", "down"])  # remaining actions
    probabilities = [
        0.4 if a == "right" else 0.3 if a == "up" else 0.15 for a in actions
    ]
    return np.random.choice(actions, p=probabilities)


def generate_episode(env, policy):
    state = (1, 1)
    episode = []
    while not env.is_terminal(state):
        action = policy(state)
        next_state, reward = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state
    return episode


def calculate_importance_sampling_ratios(episode, target_policy, behavior_policy):
    """
    Calculate the importance sampling ratios for each step in the episode.
    """
    ratios = []
    for state, action, _ in episode:
        # Probability of taking 'action' in 'state' under the target and behavior policies
        target_prob = target_policy_probability(state, action, target_policy)
        behavior_prob = behavior_policy_probability(state, action, behavior_policy)

        ratio = target_prob / behavior_prob
        ratios.append(ratio)

    return ratios


def target_policy_probability(state, action, target_policy):
    # In this simple example, we hardcode the probabilities of each action in the target policy
    if action == "right" and state[1] < 4:
        return 0.4
    if action == "up" and state[0] > 1:
        return 0.3
    return 0.15  # probability for left and down


def behavior_policy_probability(state, action, behavior_policy):
    # The behavior policy has equal probability for all actions
    return 0.25


value_estimates = {}


# 7.13 + 7.2
def off_policy_prediction_ver_1(episode, target_policy, behavior_policy):
    ratios = calculate_importance_sampling_ratios(
        episode, target_policy, behavior_policy
    )
    gamma = 0.95
    alpha = 0.95
    for i in reversed(range(len(episode))):
        state, _, reward = episode[i]
        G = ratios[i] * (reward + gamma * G) + (1 - ratios[i]) * value_estimates.get(
            state, 0
        )
        if state not in value_estimates:
            value_estimates[state] = 0
        value_estimates[state] += alpha * (G - value_estimates.get(state, 0))
    return value_estimates


# 7.1 + 7.9
def off_policy_prediction_ver_2(episode, target_policy, behavior_policy):
    pass


# def off_policy_prediction(episode, target_policy, behavior_policy):
#     ratios = calculate_importance_sampling_ratios(
#         episode, target_policy, behavior_policy
#     )
#     G = 0  # total return
#     W = 1  # importance sampling ratio
#     # value_estimates = {}  # store value estimates for each state

#     for i in reversed(range(len(episode))):
#         state, _, reward = episode[i]
#         G = reward + G  # update the return
#         W *= ratios[i]  # update the weight

#         # Only update the value estimates if the action taken is according to the target policy
#         if W == 0:
#             break
#         if state not in value_estimates:
#             value_estimates[state] = 0
#         value_estimates[state] += W * G

#     return value_estimates


if __name__ == "__main__":
    env = GridWorld()

    for _ in range(100):
        # Generate an episode using behavior policy
        episode = generate_episode(env, behavior_policy)
        print(episode)

        # Estimate the values using off-policy prediction
        value_estimates = off_policy_prediction_ver_1(
            episode, target_policy, behavior_policy
        )
        print(value_estimates)
