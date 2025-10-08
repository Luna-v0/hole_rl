from uuid import UUID
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from buraco.game import GameManager
from buraco.websockets import ConnectionManager
from buraco.models import Card
from bot_requests import (
    JoinGameRequest,
    AddBotRequest,
    CreateConfiguredGameRequest,
    CreateBotGameRequest,
)

app = FastAPI(
    title="Buraco API",
    description="API for the Buraco card game.",
    version="0.1.0",
)

# CORS Middleware
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://0.0.0.0:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In a production app, you would use a more robust singleton pattern
# or a dependency injection framework to manage these managers.
ws_manager = ConnectionManager()
game_manager = GameManager(ws_manager)


# --- API Endpoints ---


@app.post("/games", tags=["Game Management"])
def create_game_endpoint():
    """Creates a new Buraco game.

    Returns:
        The initial state of the newly created game.
    """
    return game_manager.create_game()


@app.post("/games/create_configured", tags=["Game Management"])
def create_configured_game_endpoint(request: CreateConfiguredGameRequest):
    """Creates a new Buraco game with a specific configuration.

    Returns:
        The initial state of the newly created game.
    """
    return game_manager.create_game(
        human_players=request.human_players, bot_players=request.bot_players
    )


@app.post("/games/create_and_start_bot_game", tags=["Game Management"])
async def create_and_start_bot_game_endpoint(request: CreateBotGameRequest):
    """Creates and starts a new Buraco game with only bots.

    Returns:
        The initial state of the newly created game.
    """
    # Create bot player configs
    from bot_requests import BotPlayer
    import asyncio

    bot_players = [
        BotPlayer(name=f"Bot {i+1}", algorithm="random")
        for i in range(request.bot_players)
    ]

    game = game_manager.create_game(
        human_players=0, bot_players=bot_players
    )
    game = game_manager.start_game(game.game_id)
    if not game:
        raise HTTPException(
            status_code=400,
            detail="Could not start game. Ensure 2 or 4 players have joined and it's not already started.",
        )

    # Start bot turns in the background (don't await - let them run continuously)
    if game.players[0].is_bot:
        asyncio.create_task(game_manager.run_bot_turns(game.game_id))

    return game_manager.get_game(game.game_id)


@app.get("/games/{game_id}", tags=["Game Management"])
def get_game_endpoint(game_id: UUID):
    """Retrieves the state of a specific game.

    Args:
        game_id: The UUID of the game to retrieve.

    Returns:
        The current state of the game.
    """
    game = game_manager.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


@app.post("/games/{game_id}/join", tags=["Game Management"])
async def join_game_endpoint(game_id: UUID, request: JoinGameRequest):
    """Adds a player to a game.

    Args:
        game_id: The UUID of the game to join.
        request: A request body containing the player's name.

    Returns:
        The player object if joined successfully.
    """
    player = game_manager.join_game(game_id, request.player_name)
    if not player:
        raise HTTPException(
            status_code=400,
            detail="Could not join game. It might be full or already started.",
        )

    game = game_manager.get_game(game_id)
    if game:
        await ws_manager.broadcast_game_state(game)

    return player


@app.post("/games/{game_id}/add_bot", tags=["Game Management"])
def add_bot_endpoint(game_id: UUID, request: AddBotRequest):
    """Adds a bot player to a game.

    Args:
        game_id: The UUID of the game to join.
        request: A request body containing the bot's name.

    Returns:
        The player object if joined successfully.
    """
    player = game_manager.add_bot_player(game_id, request.bot_name)
    if not player:
        raise HTTPException(
            status_code=400,
            detail="Could not add bot to game. It might be full or already started.",
        )
    return player


@app.post("/games/{game_id}/start", tags=["Game Management"])
async def start_game_endpoint(game_id: UUID):
    """Starts a game that has enough players.

    Args:
        game_id: The UUID of the game to start.

    Returns:
        The state of the game after starting.
    """
    print(f"\n{'='*60}")
    print(f"[START ENDPOINT] Request to start game: {game_id}")

    # Get current game state for debugging
    existing_game = game_manager.get_game(game_id)
    if existing_game:
        print(f"[START ENDPOINT] Current game state:")
        print(f"  - game_started: {existing_game.game_started}")
        print(f"  - players: {len(existing_game.players)}")
        print(f"  - game_over: {existing_game.game_over}")
    else:
        print(f"[START ENDPOINT] ❌ Game not found!")

    game = game_manager.start_game(game_id)

    if not game:
        print(f"[START ENDPOINT] ❌ Failed to start game")
        print(f"{'='*60}\n")
        raise HTTPException(
            status_code=400,
            detail="Could not start game. Ensure 2 players have joined and it's not already started.",
        )

    print(f"[START ENDPOINT] ✓ Game started successfully")
    print(f"  - Players: {[p.name for p in game.players]}")

    if game.players[0].is_bot:
        print(f"[START ENDPOINT] First player is bot, running bot turns...")
        try:
            await game_manager.run_bot_turns(game_id)
        except ValueError as e:
            print(f"[START ENDPOINT] ❌ Bot turn error: {e}")
            print(f"{'='*60}\n")
            raise HTTPException(status_code=400, detail=str(e))

    await ws_manager.broadcast_game_state(game)
    print(f"[START ENDPOINT] ✓ Complete - game state broadcasted")
    print(f"{'='*60}\n")

    return game


# --- WebSocket Endpoint ---


import json

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: UUID):
    """WebSocket endpoint for real-time game communication.

    Args:
        websocket: The WebSocket connection.
        game_id: The UUID of the game to connect to.
    """
    await ws_manager.connect(websocket, game_id)
    await ws_manager.broadcast(json.dumps({"type": "player_connected", "payload": f"A new player has connected to game {game_id}"}), game_id)
    try:
        while True:
            data = await websocket.receive_json()
            print(data)
            action = data.get("action")

            game = game_manager.get_game(game_id)
            if not game:
                await websocket.send_json({"error": "Game not found."})
                continue

            try:
                if action == "draw_from_deck":
                    player_id = UUID(data.get("player_id"))
                    game = game_manager.draw_card_from_deck(game_id, player_id)
                elif action == "take_discard_pile":
                    player_id = UUID(data.get("player_id"))
                    game = game_manager.take_discard_pile(game_id, player_id)
                elif action == "discard_card":
                    player_id = UUID(data.get("player_id"))
                    card_data = data.get("card")
                    card = Card(**card_data)
                    game = game_manager.discard_card(game_id, player_id, card)
                    if not game.game_over:
                        game_manager.next_turn(game)
                        await game_manager.run_bot_turns(game_id)
                elif action == "meld_cards":
                    player_id = UUID(data.get("player_id"))
                    cards_data = data.get("cards")
                    cards = [Card(**c) for c in cards_data]
                    target_meld_id = data.get("target_meld_id")
                    game = game_manager.meld_cards(game_id, player_id, cards, UUID(target_meld_id) if target_meld_id else None)
                elif action == "PLAY_BOT_TURN":
                    game_manager.play_bot_turn(game_id)
                    game = game_manager.get_game(game_id)
                else:
                    await websocket.send_json({"error": f"Unknown action: {action}"})
                    continue

                # Broadcast updated game state to all players
                await ws_manager.broadcast_game_state(game)

            except ValueError as e:
                await websocket.send_json({"error": str(e)})
            except Exception as e:
                await websocket.send_json({"error": f"An unexpected error occurred: {str(e)}"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, game_id)
        await ws_manager.broadcast(json.dumps({"type": "player_disconnected", "payload": f"A player has left game {game_id}"}), game_id)


# --- Root Endpoint ---


@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint of the API."""
    return {"message": "Welcome to the Buraco API!"}
