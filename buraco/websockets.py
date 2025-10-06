"""WebSocket connection management for Buraco."""

from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder
from typing import Dict, List
from uuid import UUID

from .models import GameState


class ConnectionManager:
    """Manages WebSocket connections for games.

    Attributes:
        active_connections (Dict[UUID, List[WebSocket]]): A dictionary mapping game_ids to a list of active WebSocket connections.
    """

    def __init__(self):
        """Initializes the ConnectionManager."""
        self.active_connections: Dict[UUID, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, game_id: UUID):
        """Accepts a new WebSocket connection for a game.

        Args:
            websocket: The WebSocket connection.
            game_id: The ID of the game to connect to.
        """
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)

    def disconnect(self, websocket: WebSocket, game_id: UUID):
        """Disconnects a WebSocket.

        Args:
            websocket: The WebSocket connection to disconnect.
            game_id: The ID of the game the WebSocket is associated with.
        """
        if game_id in self.active_connections and websocket in self.active_connections[game_id]:
            self.active_connections[game_id].remove(websocket)

    async def broadcast(self, message: str, game_id: UUID):
        """Broadcasts a message to all connected clients for a game.

        Args:
            message: The message to broadcast.
            game_id: The ID of the game to broadcast to.
        """
        if game_id in self.active_connections:
            for connection in self.active_connections[game_id]:
                await connection.send_text(message)

    async def broadcast_game_state(self, game_state: GameState):
        """Broadcasts the current game state to all connected clients for a game.

        Args:
            game_state: The current GameState object.
        """ 
        game_id = game_state.game_id
        if game_id in self.active_connections:
            json_compatible_game_state = jsonable_encoder(game_state)
            message = {
                "type": "game_state_update",
                "payload": json_compatible_game_state
            }
            for connection in self.active_connections[game_id]:
                await connection.send_json(message)
