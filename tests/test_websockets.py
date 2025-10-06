"""Tests for the ConnectionManager class."""

import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from buraco.websockets import ConnectionManager
from buraco.models import GameState

class TestConnectionManager(unittest.TestCase):
    """Tests for the ConnectionManager class."""

    def setUp(self):
        """Set up the test case."""
        self.manager = ConnectionManager()
        self.game_id = uuid4()

    def test_connect(self):
        """Test that a websocket can connect."""
        websocket = AsyncMock()
        asyncio.run(self.manager.connect(websocket, self.game_id))
        self.assertIn(self.game_id, self.manager.active_connections)
        self.assertIn(websocket, self.manager.active_connections[self.game_id])

    def test_disconnect(self):
        """Test that a websocket can disconnect."""
        websocket = AsyncMock()
        asyncio.run(self.manager.connect(websocket, self.game_id))
        self.manager.disconnect(websocket, self.game_id)
        self.assertNotIn(websocket, self.manager.active_connections[self.game_id])

    def test_broadcast(self):
        """Test that a message can be broadcast."""
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        asyncio.run(self.manager.connect(websocket1, self.game_id))
        asyncio.run(self.manager.connect(websocket2, self.game_id))

        message = "Hello, world!"
        asyncio.run(self.manager.broadcast(message, self.game_id))

        websocket1.send_text.assert_called_once_with(message)
        websocket2.send_text.assert_called_once_with(message)

    def test_broadcast_game_state(self):
        """Test that a game state can be broadcast."""
        websocket = AsyncMock()
        asyncio.run(self.manager.connect(websocket, self.game_id))

        game_state = GameState(game_id=self.game_id)
        asyncio.run(self.manager.broadcast_game_state(game_state))

        websocket.send_json.assert_called_once()

if __name__ == '__main__':
    unittest.main()
