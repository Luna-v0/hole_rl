from __future__ import annotations
"""Data models for the Buraco card game."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List
from uuid import UUID, uuid4


class Suit(Enum):
    """Represents the suit of a card."""

    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    SPADES = "Spades"
    JOKER = "Joker"


class TurnPhase(Enum):
    """Represents the current phase of a player's turn."""
    DRAW = "draw"
    MELD = "meld"
    DISCARD = "discard"

class Rank(Enum):
    """Represents the rank of a card."""

    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    JOKER = "Joker"


@dataclass(frozen=True)
class Card:
    """Represents a playing card."""

    rank: Rank
    suit: Suit


@dataclass
class Meld:
    """Represents a meld of cards."""

    meld_id: UUID = field(default_factory=uuid4)
    cards: List[Card] = field(default_factory=list)


@dataclass
class Player:
    """Represents a player in the game."""

    name: str
    player_id: UUID = field(default_factory=uuid4)
    is_bot: bool = False
    hand: List[Card] = field(default_factory=list)
    melds: List[Meld] = field(default_factory=list)
    score: int = 0


@dataclass
class GameState:
    """Represents the state of a Buraco game.

    Attributes:
        game_id (UUID): The unique identifier for the game.
        players (List[Player]): The list of players in the game.
        deck (Deck): The game deck.
        discard_pile (List[Card]): The discard pile.
        pots (List[List[Card]]): The two pots.
        current_turn_player_index (int): The index of the player whose turn it is.
        game_started (bool): Whether the game has started.
    """

    game_id: UUID = field(default_factory=uuid4)
    players: List[Player] = field(default_factory=list)
    deck: "Deck" | None = None
    discard_pile: List[Card] = field(default_factory=list)
    pots: List[List[Card]] = field(default_factory=list)
    current_turn_player_index: int = 0
    game_started: bool = False
    turn_phase: TurnPhase = TurnPhase.DRAW
    last_discard: Card | None = None
    game_over: bool = False
    scores: Dict[str, int] = field(default_factory=dict)