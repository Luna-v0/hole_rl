"""A simple bot that plays random actions in the game of Buraco."""

import random
from typing import Dict, Any, List
from buraco.models import GameState, Player, Card, Meld


class RandomBot:
    """A bot that plays random actions."""

    def __init__(self, player: Player):
        """Initializes the RandomBot.

        Args:
            player (Player): The player instance this bot will control.
        """
        self.player = player

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Gets the action from the bot based on the current game phase.

        Args:
            observation (Dict[str, Any]): The game state observation.

        Returns:
            Dict[str, Any]: The action to take.
        """
        phase = observation.get("turn_phase")

        if phase == "draw":
            return self._draw_phase(observation)
        elif phase == "meld":
            return self._meld_phase(observation)
        elif phase == "discard":
            return self._discard_phase(observation)
        else:
            # Should not happen
            return {}

    def _draw_phase(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Determines the action for the draw phase."""
        can_pick_discard = observation.get("can_pick_discard", False)

        if can_pick_discard and random.choice([True, False]):
            return {"action_type": "DRAW", "choice": "PICK_DISCARD"}
        else:
            return {"action_type": "DRAW", "choice": "DRAW_STOCK"}

    def _meld_phase(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Determines the action for the meld phase."""
        hand = observation.get("hand", [])
        melds = self.player.melds

        if not hand:
            return {"action_type": "MELD", "ops": []}

        # Try to add a card to an existing meld
        for meld in melds:
            for card in hand:
                if self._can_add_to_meld(card, meld):
                    return {"action_type": "MELD", "ops": [{"meld_cards": [card], "target_meld_id": meld.meld_id}]}

        # Find possible new melds (sequences)
        possible_melds = self._find_sequences(hand)

        if possible_melds:
            meld_to_play = random.choice(possible_melds)
            return {"action_type": "MELD", "ops": [{"meld_cards": meld_to_play}]}

        return {"action_type": "MELD", "ops": []}

    def _find_sequences(self, hand: List[Card]) -> List[List[Card]]:
        """Finds all possible sequences in the hand."""
        sequences = []
        suits = {}
        wildcards = []
        for card in hand:
            if card.rank.value in ["JOKER", "TWO"]:
                wildcards.append(card)
                continue
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card)

        for suit in suits:
            cards_in_suit = suits[suit]
            if len(cards_in_suit) + len(wildcards) < 3:
                continue

            rank_map = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13}
            cards_in_suit.sort(key=lambda c: rank_map.get(c.rank.value, 0))

            for i in range(len(cards_in_suit) - 1):
                for j in range(i + 1, len(cards_in_suit)):
                    meld = cards_in_suit[i:j+1]
                    if self._is_valid_sequence(meld + wildcards):
                        sequences.append(meld)
        return sequences

    def _can_add_to_meld(self, card: Card, meld: Meld) -> bool:
        """Checks if a card can be added to a meld."""
        return self._is_valid_sequence(meld.cards + [card])

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

    def _discard_phase(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Determines the action for the discard phase."""
        hand = observation.get("hand", [])
        if not hand:
            return {"action_type": "DISCARD", "card": None}

        card_to_discard = random.choice(hand)
        return {"action_type": "DISCARD", "card": card_to_discard}
