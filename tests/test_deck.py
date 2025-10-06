"""Tests for the Deck class."""

import unittest
from buraco.deck import Deck
from buraco.models import Rank, Suit

class TestDeck(unittest.TestCase):
    """Tests for the Deck class."""

    def setUp(self):
        """Set up the test case."""
        self.deck = Deck()

    def test_deck_size(self):
        """Test that the deck is initialized with 108 cards."""
        self.assertEqual(len(self.deck.cards), 108)

    def test_deck_shuffle(self):
        """Test that the deck is shuffled."""
        original_deck = self.deck.cards[:]
        self.deck.shuffle()
        self.assertNotEqual(original_deck, self.deck.cards)

    def test_deck_draw(self):
        """Test that cards can be drawn from the deck."""
        drawn_cards = self.deck.draw(5)
        self.assertEqual(len(drawn_cards), 5)
        self.assertEqual(len(self.deck.cards), 103)

    def test_deck_draw_too_many(self):
        """Test that an exception is raised when trying to draw more cards than are in the deck."""
        with self.assertRaises(ValueError):
            self.deck.draw(109)

    def test_deck_deal(self):
        """Test that the deal method returns the correct number of cards for each player and the pots."""
        player1_hand, player2_hand, pot1, pot2 = self.deck.deal()
        self.assertEqual(len(player1_hand), 11)
        self.assertEqual(len(player2_hand), 11)
        self.assertEqual(len(pot1), 11)
        self.assertEqual(len(pot2), 11)
        self.assertEqual(len(self.deck.cards), 108 - 44)

if __name__ == '__main__':
    unittest.main()
