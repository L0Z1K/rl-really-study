import itertools
import random


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


if __name__ == "__main__":
    game = BlackJack()

    agent = MonteCarlo()

    for epoch in range(1000):
        success_rate = 0
        for _ in range(10000):
            dealer, player = game.draw_cards()
            dealer_first_card = dealer[0]

            histories = []
            while True:
                count_aces = sum([1 if card[1] == "A" else 0 for card in player])
                state = (
                    count_aces,
                    game.calculate_card_value(player),
                    game.calculate_card_value([dealer_first_card]),
                )
                action = agent.get_action(
                    state,
                    ["hit", "stand"],
                )
                histories.append((state, action))
                if action == "stand":
                    break
                else:
                    player.append(game.cards.pop())
                    if game.calculate_card_value(player) > 21:
                        break

            if game.check_winner(player, dealer):
                reward = 1
            else:
                reward = -1

            for state, action in histories:
                agent.update_Q(state, action, reward)

            success_rate += reward

        success_rate = (success_rate + 10000) / 20000
        print(f"Epoch: {epoch+1}, Success rate: {success_rate:.2%}")
        agent.store_Q("Q.txt")
