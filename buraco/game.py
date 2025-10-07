"""Game management for Buraco."""

from typing import Any, Dict, List
from uuid import UUID

from bot_requests import BotConfig
from .bots.bot_factory import BotFactory
from .deck import Deck
from .models import Card, GameState, Meld, Player, TurnPhase


class GameManager:
    """Manages active Buraco games.

    Attributes:
        games (Dict[UUID, GameState]): A dictionary of active games, keyed by game_id.
    """

    def __init__(self):
        """Initializes the GameManager."""
        self.games: Dict[UUID, GameState] = {}
        self.bots: Dict[UUID, Dict[UUID, Any]] = {}
        self.bot_factory = BotFactory()

    def create_game(self, human_players: int = 0, bot_players: List[BotConfig] = []) -> GameState:
        """Creates a new, empty game.

        Returns:
            GameState: The newly created game state.
        """
        game_state = GameState()
        game_state.deck = Deck()
        self.games[game_state.game_id] = game_state
        self.bots[game_state.game_id] = {}

        for _ in range(human_players):
            self.join_game(game_state.game_id, f"Human Player {len(game_state.players) + 1}")

        for bot_config in bot_players:
            self.add_bot_player(game_state.game_id, bot_config.name, bot_config.algorithm)

        return game_state



    def get_game(self, game_id: UUID) -> GameState | None:
        """Gets the state of a specific game.

        Args:
            game_id: The ID of the game to get.

        Returns:
            The game state if found, otherwise None.
        """
        return self.games.get(game_id)

    def join_game(self, game_id: UUID, player_name: str) -> Player | None:
        """Adds a player to a game.

        Args:
            game_id: The ID of the game to join.
            player_name: The name of the player joining.

        Returns:
            The newly created player if joined successfully, otherwise None.
        """
        game = self.get_game(game_id)
        if not game or game.game_started:
            return None

        if len(game.players) < 4:  # Allow up to 4 players
            new_player = Player(name=player_name)
            game.players.append(new_player)
            return new_player
        return None

    def add_bot_player(
        self, game_id: UUID, bot_name: str, algorithm: str = "random"
    ) -> Player | None:
        """Adds a bot player to a game.

        Args:
            game_id: The ID of the game to join.
            bot_name: The name of the bot.
            algorithm: The algorithm the bot will use.

        Returns:
            The newly created player if joined successfully, otherwise None.
        """
        game = self.get_game(game_id)
        if not game or game.game_started:
            return None

        if len(game.players) < 4:  # Allow up to 4 players
            new_player = Player(name=bot_name, is_bot=True)
            game.players.append(new_player)
            bot_instance = self.bot_factory.create_bot(algorithm, new_player)
            self.bots[game_id][new_player.player_id] = bot_instance
            return new_player
        return None



    def play_bot_turn(self, game_id: UUID):
        """Plays a single turn for a bot player."""
        game = self.get_game(game_id)
        if not game or not game.game_started:
            return

        player = self._get_current_player(game)
        if not player.is_bot:
            return

        bot_instance = self.bots[game_id][player.player_id]

        # Draw Phase
        game.turn_phase = TurnPhase.DRAW
        observation = self._get_observation(game, player)
        action = bot_instance.get_action(observation)
        if action.get("choice") == "PICK_DISCARD":
            self.take_discard_pile(game_id, player.player_id)
        else:
            self.draw_card_from_deck(game_id, player.player_id)

        # Meld Phase
        game.turn_phase = TurnPhase.MELD
        observation = self._get_observation(game, player)
        action = bot_instance.get_action(observation)
        if action.get("ops"):
            for op in action["ops"]:
                if op.get("meld_cards"):
                    game = self.meld_cards(game_id, player.player_id, op["meld_cards"])

        # Discard Phase
        game.turn_phase = TurnPhase.DISCARD
        observation = self._get_observation(game, player)
        action = bot_instance.get_action(observation)
        if action.get("card"):
            self.discard_card(game_id, player.player_id, action["card"])

        self.next_turn(game)

    def _get_observation(self, game_state: GameState, player: Player) -> Dict:
        """Generates the observation for a player."""
        return {
            "hand": player.hand,
            "discard_pile": game_state.discard_pile,
            "turn_phase": game_state.turn_phase.value,
            "can_pick_discard": len(game_state.discard_pile) > 0,
        }

    def start_game(self, game_id: UUID) -> GameState | None:
        """Starts a game, deals cards, and prepares the discard pile.

        Args:
            game_id: The ID of the game to start.

        Returns:
            The updated game state if started, otherwise None.
        """
        game = self.get_game(game_id)
        if not game or game.game_started or len(game.players) not in [2, 4]:
            return None

        game.game_started = True

        # Deal hands and pots
        num_players = len(game.players)
        dealt_cards = game.deck.deal(num_players)
        for i in range(num_players):
            game.players[i].hand = dealt_cards[i]
        game.pots = dealt_cards[num_players:]

        # Create discard pile with one card
        if game.deck.cards:
            game.discard_pile.append(game.deck.draw()[0])

        return game

    def _get_player(self, game_state: GameState, player_id: UUID) -> Player:
        """Helper to get a player by ID."""
        for player in game_state.players:
            if player.player_id == player_id:
                return player
        raise ValueError("Player not found in game.")

    def _get_current_player(self, game_state: GameState) -> Player:
        """Helper to get the current player."""
        return game_state.players[game_state.current_turn_player_index]

    def next_turn(self, game_state: GameState):
        """Advances to the next player's turn."""
        game_state.current_turn_player_index = (game_state.current_turn_player_index + 1) % len(game_state.players)
        game_state.turn_phase = TurnPhase.DRAW

    def draw_card_from_deck(self, game_id: UUID, player_id: UUID) -> GameState:
        """Player draws a card from the deck."""
        game = self.get_game(game_id)
        if not game:
            raise ValueError("Game not found.")

        current_player = self._get_current_player(game)
        if current_player.player_id != player_id:
            raise ValueError("It's not this player's turn.")
        if game.turn_phase != TurnPhase.DRAW:
            raise ValueError("Cannot draw at this phase of the turn.")

        if len(current_player.hand) == 0 and game.pots:
            current_player.hand.extend(game.pots.pop(0))
            team_index = game.players.index(current_player) % 2
            game.pot_taken_by_team[team_index] = True
            game.turn_phase = TurnPhase.MELD
            return game

        if not game.deck.cards:
            if game.pots:
                game.deck.cards = game.pots.pop(0)
                game.deck.shuffle()
            else:
                game.game_over = True
                self._calculate_scores(game)
                return game

        drawn_card = game.deck.draw(1)[0]
        current_player.hand.append(drawn_card)
        game.turn_phase = TurnPhase.MELD
        return game

    def take_discard_pile(self, game_id: UUID, player_id: UUID) -> GameState:
        """Player takes the entire discard pile."""
        game = self.get_game(game_id)
        if not game:
            raise ValueError("Game not found.")

        current_player = self._get_current_player(game)
        if current_player.player_id != player_id:
            raise ValueError("It's not this player's turn.")
        if game.turn_phase != TurnPhase.DRAW:
            raise ValueError("Cannot take discard pile at this phase of the turn.")
        if not game.discard_pile:
            raise ValueError("Discard pile is empty.")

        # Add all cards from discard pile to player's hand
        current_player.hand.extend(game.discard_pile)
        game.discard_pile = []  # Clear the discard pile
        game.turn_phase = TurnPhase.MELD
        return game

    def discard_card(self, game_id: UUID, player_id: UUID, card: Card) -> GameState:
        """Player discards a card to end their turn."""
        game = self.get_game(game_id)
        if not game:
            raise ValueError("Game not found.")

        current_player = self._get_current_player(game)
        if current_player.player_id != player_id:
            raise ValueError("It's not this player's turn.")
        if game.turn_phase != TurnPhase.DISCARD:
            raise ValueError("Cannot discard at this phase of the turn.")
        if card not in current_player.hand:
            raise ValueError("Card not in player's hand.")
        if len(current_player.hand) == 1 and card.rank.value in ["JOKER", "TWO"]:
            raise ValueError("Cannot discard a wildcard to end the game.")

        current_player.hand.remove(card)
        game.discard_pile.append(card)
        game.last_discard = card
        self._check_game_over(game)
        return game

    def meld_cards(self, game_id: UUID, player_id: UUID, cards: List[Card], target_meld_id: UUID | None = None) -> GameState:
        """Player creates a new meld or adds to an existing one."""
        game = self.get_game(game_id)
        if not game:
            raise ValueError("Game not found.")

        current_player = self._get_current_player(game)
        if current_player.player_id != player_id:
            raise ValueError("It's not this player's turn.")
        if game.turn_phase not in [TurnPhase.MELD, TurnPhase.DISCARD]:
            raise ValueError("Cannot meld at this phase of the turn.")

        # Basic validation: all cards must be in player's hand
        if not all(c in current_player.hand for c in cards):
            raise ValueError("One or more cards not in player's hand.")

        if not self._is_valid_sequence(cards):
            raise ValueError("Invalid meld. Only sequences are allowed.")

        if target_meld_id:
            # Add to existing meld
            target_meld = None
            team_index = game.players.index(current_player) % 2
            for i, player in enumerate(game.players):
                if i % 2 == team_index:
                    for meld in player.melds:
                        if meld.meld_id == target_meld_id:
                            target_meld = meld
                            break
                    if target_meld:
                        break

            if not target_meld:
                raise ValueError("Target meld not found.")
            target_meld.cards.extend(cards)
        else:
            # Create new meld
            new_meld = Meld(cards=cards)
            current_player.melds.append(new_meld)

        for card in cards:
            current_player.hand.remove(card)

        # After melding, player can still meld or discard
        game.turn_phase = TurnPhase.DISCARD
        return game

    def _is_valid_sequence(self, cards: List[Card]) -> bool:
        """Checks if a list of cards is a valid sequence."""
        if len(cards) < 3:
            return False

        non_wild_cards = [card for card in cards if card.rank.value not in ["JOKER", "TWO"]]
        if not non_wild_cards:
            return False  # a meld must have at least one natural card

        suit = non_wild_cards[0].suit
        if not all(card.suit == suit or card.rank.value in ["JOKER", "TWO"] for card in non_wild_cards):
            return False

        # Sort cards by rank
        rank_map = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13}
        cards.sort(key=lambda c: rank_map.get(c.rank.value, 0))

        for i in range(len(cards) - 1):
            rank1 = cards[i].rank.value
            rank2 = cards[i+1].rank.value

            if rank1 == "JOKER" or rank1 == "TWO":
                continue

            if rank2 == "JOKER" or rank2 == "TWO":
                continue

            if rank_map[rank2] - rank_map[rank1] != 1:
                # Handle Ace at the end
                if rank1 == "A" and rank2 == "K":
                    continue
                return False

        return True

    def _check_game_over(self, game: GameState):
        """Checks if the game is over and calculates the scores."""
        if not game.deck.cards and not game.pots:
            game.game_over = True
            self._calculate_scores(game)
            return

        current_player = self._get_current_player(game)
        if len(current_player.hand) == 0:
            # Check if the player's team has a buraco and has taken the pot
            team_index = game.players.index(current_player) % 2
            if game.pot_taken_by_team[team_index]:
                team_has_buraco = False
                for i, player in enumerate(game.players):
                    if i % 2 == team_index:
                        for meld in player.melds:
                            if len(meld.cards) >= 7:
                                team_has_buraco = True
                                break
                        if team_has_buraco:
                            break
                
                if team_has_buraco:
                    game.game_over = True
                    self._calculate_scores(game)

    def _calculate_scores(self, game: GameState):
        """Calculates the scores for each team."""
        scores = {"team1": 0, "team2": 0}
        closing_team = game.players.index(self._get_current_player(game)) % 2

        for i, player in enumerate(game.players):
            team = "team1" if i % 2 == 0 else "team2"
            for meld in player.melds:
                is_dirty = any(card.rank.value in ["JOKER", "TWO"] for card in meld.cards)
                if len(meld.cards) >= 7:
                    scores[team] += 200 if not is_dirty else 100
                for card in meld.cards:
                    if card.rank.value == "JOKER":
                        scores[team] += 30
                    elif card.rank.value == "TWO":
                        scores[team] += 20
                    elif card.rank.value == "ACE":
                        scores[team] += 15
                    elif card.rank.value in ["KING", "QUEEN", "JACK", "TEN", "NINE", "EIGHT"]:
                        scores[team] += 10
                    else:
                        scores[team] += 5
            for card in player.hand:
                if card.rank.value == "JOKER":
                    scores[team] -= 30
                elif card.rank.value == "TWO":
                    scores[team] -= 20
                elif card.rank.value == "ACE":
                    scores[team] -= 15
                elif card.rank.value in ["KING", "QUEEN", "JACK", "TEN", "NINE", "EIGHT"]:
                    scores[team] -= 10
                else:
                    scores[team] -= 5
        
        scores["team1" if closing_team == 0 else "team2"] += 100
        game.scores = scores
        print(game.scores)
