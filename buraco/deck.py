"""Deck management for the Buraco card game."""

import random
from typing import List

from .models import Card, Rank, Suit


class Deck:
    """Represents the deck of cards for the game.

    Attributes:
        cards (List[Card]): The list of cards in the deck.
    """

    def __init__(self):
        """Initializes the deck with two standard decks and four jokers."""
        self.cards: List[Card] = []
        self.build()

    def build(self):
        """Builds the deck with 108 cards (2 standard decks + 4 jokers)."""
        self.cards = []
        for _ in range(2):  # Two decks
            for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
                for rank in [r for r in Rank if r != Rank.JOKER]:
                    self.cards.append(Card(rank=rank, suit=suit))
        for _ in range(4):  # Four jokers
            self.cards.append(Card(rank=Rank.JOKER, suit=Suit.JOKER))

    def shuffle(self):
        """Shuffles the deck."""
        random.shuffle(self.cards)

    def draw(self, count: int = 1) -> List[Card]:
        """Draws a specified number of cards from the deck.

        Args:
            count (int): The number of cards to draw.

        Returns:
            List[Card]: A list of cards drawn from the deck.
        
        Raises:
            ValueError: If there are not enough cards in the deck to draw.
        """
        if len(self.cards) < count:
            raise ValueError("Not enough cards in the deck to draw.")
        drawn_cards = self.cards[:count]
        self.cards = self.cards[count:]
        return drawn_cards

    def deal(self, num_players: int = 2) -> List[List[Card]]:
        """Deals the cards for a new game.

        Args:
            num_players (int): The number of players in the game (2 or 4).

        Returns:
            List[List[Card]]: A list of lists, where each inner list represents a
                hand or a pot.
        """
        if num_players not in [2, 4]:
            raise ValueError("Number of players must be 2 or 4.")

        self.shuffle()
        hands = [self.draw(11) for _ in range(num_players)]
        pots = [self.draw(11) for _ in range(2)]
        return hands + pots
