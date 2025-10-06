## Backend Game Logic Implementation

This document outlines the plan for implementing the backend logic for the Buraco card game.

## Use the ../INSTRUCTIONS.md for the rules of the game.

### Core Components

- **Game State:** A dataclass to represent the entire state of the game, including the deck, discard pile, pots, players, and whose turn it is.
- **Player:** A dataclass to represent a player, including their hand, melds, and score.
- **Card:** A dataclass to represent a card, with suit and rank.
- **Meld:** A dataclass to represent a meld of cards. It is recommended to add a unique `id` to this dataclass to make it easier to reference from the frontend.
- **Deck:** A class to manage the deck of cards, including shuffling and drawing.

### Game Flow

The game flow will be managed by a `Game` class that will handle the following:

- **Initialization:** Setting up the game with two decks of cards, shuffling, dealing, and creating the pots.
- **Turns:** Managing the turn-based gameplay, including drawing, melding, and discarding.
- **Rules Enforcement:** Ensuring that all moves made by players are valid according to the rules of Buraco.
- **Scoring:** Calculating the scores at the end of each round.

### WebSockets

WebSockets will be used for real-time communication between the server and the clients. The backend should handle the following incoming messages from the client:

- **Create a new meld:**
  ```json
  {
    "action": "CREATE_MELD",
    "payload": {
      "cards": [
        { "rank": "KING", "suit": "HEARTS" },
        { "rank": "QUEEN", "suit": "HEARTS" },
        { "rank": "JACK", "suit": "HEARTS" }
      ]
    }
  }
  ```

- **Add a card to an existing meld:**
  ```json
  {
    "action": "ADD_TO_MELD",
    "payload": {
      "meld_id": 0, // or a unique meld ID
      "card": { "rank": "ACE", "suit": "SPADES" }
    }
  }
  ```

- **Draw a card from the deck:**
  ```json
  {
    "action": "DRAW_FROM_DECK"
  }
  ```

- **Take the discard pile:**
  ```json
  {
    "action": "TAKE_DISCARD_PILE"
  }
  ```

- **Discard a card:**
  ```json
  {
    "action": "DISCARD_CARD",
    "payload": {
      "card": { "rank": "ACE", "suit": "SPADES" }
    }
  }
  ```

After each action, the backend should broadcast the updated `GameState` to all clients in the game.

The following events will be handled:

- **`player_joined`:** A new player has joined the game.
- **`game_started`:** The game has started.
- **`card_drawn`:** A player has drawn a card.
- **`card_melded`:** A player has melded cards.
- **`card_discarded`:** A player has discarded a card.
- **`pot_taken`:** A player has taken the pot.
- **`round_ended`:** The round has ended.
- **`game_ended`:** The game has ended.

### API Endpoints

In addition to the WebSockets, the following RESTful API endpoints will be created:

- **`POST /games`:** Create a new game.
- **`GET /games/{game_id}`:** Get the state of a game.
- **`POST /games/{game_id}/join`:** Join a game.
