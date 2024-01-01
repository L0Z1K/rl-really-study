from collections import namedtuple
import itertools
import random
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class BlackJack:
    def __init__(self):
        self.cards = self.init_cards()

    def start(self):
        dealer, player = self.draw_cards()
        self.show_dealer_cards(dealer)
        self.show_player_cards(player)

        while True:
            choice = input("Do you want to hit or stand? (h/s) ")
            if choice == "h":
                player.append(self.cards.pop())
                self.show_player_cards(player)
                if self.calculate_card_value(player) > 21:
                    print("You lose!")
                    break
            elif choice == "s":
                break
            else:
                print("Invalid input!")

        self.dealer_play(dealer)
        print(
            f"Dealer's cards: {', '.join([f'{card[0]} {card[1]}' for card in dealer])}"
        )
        if self.check_winner(player, dealer):
            print("You win!")
        else:
            print("You lose!")

    def init_cards(self):
        cards = itertools.product(
            ["♠︎", "♥︎", "♦︎", "♣︎"],
            ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10],
        )
        cards = list(cards)
        random.shuffle(cards)
        return cards

    def draw_cards(self):
        if len(self.cards) < 10:
            self.cards = self.init_cards()
        dealer = []
        player = []
        for i in range(2):
            dealer.append(self.cards.pop())
            player.append(self.cards.pop())
        return dealer, player

    def show_dealer_cards(self, dealer):
        first_card = dealer[0]
        print(f"Dealer's first card: {first_card[0]} {first_card[1]}")

    def show_player_cards(self, player):
        print(f"Your cards: {', '.join([f'{card[0]} {card[1]}' for card in player])}")

    def dealer_play(self, dealer):
        while True:
            if self.calculate_card_value(dealer) >= 17:
                break
            else:
                dealer.append(self.cards.pop())

    @staticmethod
    def calculate_card_value(cards, option="player"):
        total = 0
        for card in cards:
            if card[1] == "A":
                if option == "dealer":
                    total += 11
                elif total >= 11:
                    total += 1
                else:
                    total += 11
            else:
                total += card[1]
        return total

    def check_winner(self, player, dealer):
        if self.calculate_card_value(player) > 21:
            return False
        elif self.calculate_card_value(dealer) > 21:
            return True
        elif self.calculate_card_value(player) >= self.calculate_card_value(dealer):
            return True
        else:
            return False


class MonteCarlo:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.Q = {}

    def get_Q(self, state, action):
        if (state, action) not in self.Q:
            self.Q[(state, action)] = random.random() / 100
        return self.Q[(state, action)]

    def update_Q(self, state, action, reward):
        self.Q[(state, action)] += self.alpha * (reward - self.Q[(state, action)])

    def store_Q(self, filename):
        with open(filename, "w") as f:
            for (state, action), value in self.Q.items():
                f.write(f"{state}\t{action}\t{value}\n")

    def load_Q(self, filename):
        with open(filename, "r") as f:
            for line in f:
                state, action, value = line.strip().split("\t")
                self.Q[(eval(state), action)] = float(value)

    def get_action(self, state, actions):
        Qs = [self.get_Q(state, action) for action in actions]
        return actions[Qs.index(max(Qs))]  # greedy


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(3, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        action_prob = F.softmax(self.action_head(x), dim=-1)

        state_values = self.value_head(x)

        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + 0.99 * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    srs = []
    for i_episode in count(1):
        game = BlackJack()
        success_rate = 0
        for _ in range(300):
            dealer, player = game.draw_cards()
            dealer_first_card = dealer[0]

            cnt = 0
            while True:
                count_aces = sum([1 if card[1] == "A" else 0 for card in player])
                state = np.array(
                    [
                        count_aces,
                        game.calculate_card_value(player) / 21,
                        game.calculate_card_value([dealer_first_card]) / 21,
                    ]
                )
                action = select_action(state)
                cnt += 1
                if action == 0:
                    break
                elif action == 1:
                    player.append(game.cards.pop())
                    if game.calculate_card_value(player) > 21:
                        break
                else:
                    raise ValueError("Invalid action!", action)

            if game.check_winner(player, dealer):
                reward = 1
            else:
                reward = -1
            model.rewards += [reward] * cnt
            success_rate += reward
        finish_episode()
        success_rate = (success_rate + 300) / 600
        srs.append(success_rate)
        print(f"Episode: {i_episode}, Success rate: {success_rate:.2%}")
        if i_episode == 200:
            # save the model
            torch.save(model.state_dict(), "./model.pt")
            break

    # plot the success rate and cumulative success rate
    import matplotlib.pyplot as plt

    plt.plot(srs)
    plt.xlabel("Episode")
    plt.ylabel("Success rate")
    plt.savefig("./success_rate.png")
    plt.clf()
    plt.plot(np.cumsum(srs) / np.arange(1, len(srs) + 1))
    plt.xlabel("Episode")
    plt.ylabel("Cumulative success rate")
    plt.savefig("./cumulative_success_rate.png")


if __name__ == "__main__":
    main()
