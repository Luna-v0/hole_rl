"""Tests for the data models."""

import unittest
from uuid import uuid4
from buraco.models import Card, Meld, Player, GameState, Rank, Suit, TurnPhase

class TestModels(unittest.TestCase):
    """Tests for the data models."""

    def test_card(self):
        """Test the Card dataclass."""
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        self.assertEqual(card.rank, Rank.ACE)
        self.assertEqual(card.suit, Suit.SPADES)

    def test_meld(self):
        """Test the Meld dataclass."""
        card1 = Card(rank=Rank.ACE, suit=Suit.SPADES)
        card2 = Card(rank=Rank.TWO, suit=Suit.SPADES)
        meld = Meld(cards=[card1, card2])
        self.assertEqual(len(meld.cards), 2)
        self.assertEqual(meld.cards[0], card1)
        self.assertEqual(meld.cards[1], card2)

    def test_player(self):
        """Test the Player dataclass."""
        player = Player(name="Test Player")
        self.assertEqual(player.name, "Test Player")
        self.assertIsNotNone(player.player_id)
        self.assertEqual(player.hand, [])
        self.assertEqual(player.melds, [])
        self.assertEqual(player.score, 0)

    def test_game_state(self):
        """Test the GameState dataclass."""
        game_state = GameState()
        self.assertIsNotNone(game_state.game_id)
        self.assertEqual(game_state.players, [])
        self.assertEqual(game_state.discard_pile, [])
        self.assertEqual(game_state.pots, [])
        self.assertEqual(game_state.current_turn_player_index, 0)
        self.assertFalse(game_state.game_started)
        self.assertEqual(game_state.turn_phase, TurnPhase.DRAW)
        self.assertIsNone(game_state.last_discard)

if __name__ == '__main__':
    unittest.main()
