import pytest
from fastapi.testclient import TestClient
import json
from main import app

client = TestClient(app)

@pytest.mark.anyio
async def test_full_bot_game():
    """Tests a full game with only bots."""
    # Create a new game with 4 bots
    response = client.post(
        "/games/create_configured",
        json={"human_players": 0, "bot_players": [{"name": "Bot 1", "algorithm": "random"}, {"name": "Bot 2", "algorithm": "random"}, {"name": "Bot 3", "algorithm": "random"}, {"name": "Bot 4", "algorithm": "random"}]},
    )
    assert response.status_code == 200
    game_data = response.json()
    game_id = game_data["game_id"]

    # Start the game
    response = client.post(f"/games/{game_id}/start")
    assert response.status_code == 200

    # Connect to the WebSocket
    with client.websocket_connect(f"/ws/{game_id}") as websocket:
        while True:
            message = websocket.receive_json()
            if message.get("payload", {}).get("game_over"):
                break

    # Get the final game state
    response = client.get(f"/games/{game_id}")
    assert response.status_code == 200
    game_data = response.json()
    assert game_data["game_over"]
