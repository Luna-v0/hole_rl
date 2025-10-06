"""Tests for the game logic."""

import unittest
from uuid import uuid4

from buraco.game import GameManager


class TestGame(unittest.TestCase):
    """Test suite for the game logic."""

    def setUp(self):
        """Set up for the tests."""
        self.game_manager = GameManager()

    def test_create_game(self):
        """Test that a game can be created."""
        game = self.game_manager.create_game()
        self.assertIsNotNone(game)
        self.assertEqual(len(game.players), 0)

    def test_join_game(self):
        """Test that a player can join a game."""
        game = self.game_manager.create_game()
        player = self.game_manager.join_game(game.game_id, "Test Player")
        self.assertIsNotNone(player)
        self.assertEqual(len(game.players), 1)

    def test_start_game(self):
        """Test that a game can be started."""
        game = self.game_manager.create_game()
        self.game_manager.join_game(game.game_id, "Player 1")
        self.game_manager.join_game(game.game_id, "Player 2")
        started_game = self.game_manager.start_game(game.game_id)
        self.assertTrue(started_game.game_started)
        self.assertEqual(len(started_game.players[0].hand), 11)
        self.assertEqual(len(started_game.players[1].hand), 11)
        self.assertEqual(len(started_game.discard_pile), 1)


if __name__ == "__main__":
    unittest.main()