from pydantic import BaseModel
from typing import List
from uuid import UUID
from buraco.models import Card

# Pydantic models for request bodies
class JoinGameRequest(BaseModel):
    """Request model for a player joining a game."""

    player_name: str


class AddBotRequest(BaseModel):
    """Request model for adding a bot to a game."""

    bot_name: str


class BotConfig(BaseModel):
    """Request model for bot configuration."""

    name: str
    algorithm: str


class CreateConfiguredGameRequest(BaseModel):
    """Request model for creating a configured game."""

    human_players: int = 0
    bot_players: List[BotConfig] = []


class CreateBotGameRequest(BaseModel):
    """Request model for creating a bot-only game."""

    bot_players: List[BotConfig] = []


class DrawFromDeckRequest(BaseModel):
    """Request model for a player drawing a card from the deck."""

    player_id: UUID


class TakeDiscardPileRequest(BaseModel):
    """Request model for a player taking the discard pile."""

    player_id: UUID


class DiscardCardRequest(BaseModel):
    """Request model for a player discarding a card."""

    player_id: UUID
    card: Card


class MeldCardsRequest(BaseModel):
    """Request model for a player melding cards."""

    player_id: UUID
    cards: List[Card]
    target_meld_id: UUID | None = None
