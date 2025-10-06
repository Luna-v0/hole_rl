"""Tests for the FastAPI application."""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from uuid import uuid4

from main import app
from buraco.models import Rank, Suit

class TestMain(unittest.TestCase):
    """Tests for the FastAPI application."""

    def setUp(self):
        """Set up the test case."""
        self.client = TestClient(app)

    def test_read_root(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Welcome to the Buraco API!"})

    def test_create_game_endpoint(self):
        """Test the create game endpoint."""
        response = self.client.post("/games")
        self.assertEqual(response.status_code, 200)
        self.assertIn("game_id", response.json())

    def test_get_game_endpoint(self):
        """Test the get game endpoint."""
        create_response = self.client.post("/games")
        game_id = create_response.json()["game_id"]
        get_response = self.client.get(f"/games/{game_id}")
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(get_response.json()["game_id"], game_id)

    def test_join_game_endpoint(self):
        """Test the join game endpoint."""
        create_response = self.client.post("/games")
        game_id = create_response.json()["game_id"]
        join_response = self.client.post(f"/games/{game_id}/join", json={"player_name": "Test Player"})
        self.assertEqual(join_response.status_code, 200)
        self.assertEqual(join_response.json()["name"], "Test Player")

    def test_start_game_endpoint(self):
        """Test the start game endpoint."""
        create_response = self.client.post("/games")
        game_id = create_response.json()["game_id"]
        self.client.post(f"/games/{game_id}/join", json={"player_name": "Player 1"})
        self.client.post(f"/games/{game_id}/join", json={"player_name": "Player 2"})
        start_response = self.client.post(f"/games/{game_id}/start")
        self.assertEqual(start_response.status_code, 200)
        self.assertTrue(start_response.json()["game_started"])

    def test_websocket_endpoint(self):
        """Test the websocket endpoint."""
        create_response = self.client.post("/games")
        game_id = create_response.json()["game_id"]
        player1_response = self.client.post(f"/games/{game_id}/join", json={"player_name": "Player 1"})
        player1_id = player1_response.json()["player_id"]
        self.client.post(f"/games/{game_id}/join", json={"player_name": "Player 2"})
        self.client.post(f"/games/{game_id}/start")

        with self.client.websocket_connect(f"/ws/{game_id}") as websocket:
            # Receive the initial connection message
            initial_message = websocket.receive_text()
            self.assertIn("A new player has connected", initial_message)

            # Test draw from deck
            websocket.send_json({"action": "draw_from_deck", "player_id": player1_id})
            data = websocket.receive_json()
            self.assertIn("game_id", data["payload"])

if __name__ == '__main__':
    unittest.main()
