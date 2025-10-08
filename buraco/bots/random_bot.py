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
        """Determines the action for the meld phase.

        Note: Only sequences are allowed, not sets (per game rules).
        """
        hand = observation.get("hand", [])
        melds = self.player.melds

        if not hand:
            return {"action_type": "MELD", "ops": []}

        # Try to add a card to an existing meld
        for meld in melds:
            for card in hand:
                if self._can_add_to_meld(card, meld):
                    return {"action_type": "MELD", "ops": [{"meld_cards": [card], "target_meld_id": meld.meld_id}]}

        # Find possible sequences (same suit, consecutive ranks)
        from collections import defaultdict
        suits = defaultdict(list)
        wildcards = []

        for card in hand:
            if card.rank.value in ["JOKER", "TWO"]:
                wildcards.append(card)
            else:
                suits[card.suit].append(card)

        # Try to find a sequence in each suit
        rank_map = {"A": 1, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13}

        for suit, cards_in_suit in suits.items():
            if len(cards_in_suit) < 2:
                continue

            # Sort by rank
            sorted_cards = sorted(cards_in_suit, key=lambda c: rank_map.get(c.rank.value, 0))

            # Try to form a sequence with these cards + wildcards if needed
            for i in range(len(sorted_cards)):
                for j in range(i + 1, min(i + 7, len(sorted_cards) + 1)):  # Max sequence of 7
                    potential_seq = sorted_cards[i:j]

                    # Check if we can make a valid sequence with wildcards
                    if len(potential_seq) >= 2 and wildcards:
                        # Add one wildcard and check
                        test_meld = potential_seq + [wildcards[0]]
                        if self._is_valid_meld(test_meld):
                            return {"action_type": "MELD", "ops": [{"meld_cards": test_meld}]}

                    # Check if natural sequence is valid
                    if len(potential_seq) >= 3 and self._is_valid_meld(potential_seq):
                        return {"action_type": "MELD", "ops": [{"meld_cards": potential_seq}]}

        # No valid melds found
        return {"action_type": "MELD", "ops": []}


    def _can_add_to_meld(self, card: Card, meld: Meld) -> bool:
        """Checks if a card can be added to a meld.

        A card can be added if:
        1. It's a wildcard (JOKER or TWO), OR
        2. It's the same suit as the meld AND doesn't duplicate an existing rank
        """
        # Wildcards can always be added (if the result is valid)
        if card.rank.value in ["JOKER", "TWO"]:
            return self._is_valid_meld(meld.cards + [card])

        # For non-wildcards, check if it's the same suit as the meld
        non_wild_in_meld = [c for c in meld.cards if c.rank.value not in ["JOKER", "TWO"]]
        if not non_wild_in_meld:
            # Meld only has wildcards, card must establish the suit
            return self._is_valid_meld(meld.cards + [card])

        meld_suit = non_wild_in_meld[0].suit
        if card.suit != meld_suit:
            return False

        # Check if this rank already exists in the meld (no duplicates allowed)
        existing_ranks = [c.rank.value for c in non_wild_in_meld]
        if card.rank.value in existing_ranks:
            return False

        # Finally, check if adding this card keeps it a valid sequence
        return self._is_valid_meld(meld.cards + [card])

    def _is_valid_meld(self, cards: List[Card]) -> bool:
        """Checks if a list of cards is a valid meld (sequence only, no sets).

        Per game rules: Melds must be same suit in sequential order.
        """
        if len(cards) < 3:
            return False

        non_wild_cards = [card for card in cards if card.rank.value not in ["JOKER", "TWO"]]
        if not non_wild_cards:
            return False  # a meld must have at least one natural card

        # All non-wild cards must be the same suit (required for sequences)
        first_suit = non_wild_cards[0].suit
        if not all(c.suit == first_suit for c in non_wild_cards):
            return False

        # Sort cards by rank
        rank_map = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13}
        
        sorted_cards = sorted(cards, key=lambda c: rank_map.get(c.rank.value, 0) if c.rank.value not in ["JOKER", "TWO"] else 0)

        wildcard_count = len(cards) - len(non_wild_cards)

        # Find the ranks of natural cards
        natural_ranks = sorted([rank_map[c.rank.value] for c in non_wild_cards])

        # Handle Ace-high (A,K,Q) and Ace-low (A,2,3) sequences
        is_ace_low = 1 in natural_ranks
        is_ace_high = 1 in natural_ranks and 13 in natural_ranks

        if is_ace_high:
            # Check if it could be A, K, Q ...
            # To handle this, we can temporarily treat Ace as rank 14
            if natural_ranks[0] == 1 and natural_ranks[-1] == 13:
                # check if it is a valid sequence with ace as 1
                gaps = 0
                for i in range(len(natural_ranks) - 1):
                    gaps += natural_ranks[i+1] - natural_ranks[i] - 1
                if gaps > wildcard_count:
                    # if not, check with ace as 14
                    natural_ranks.remove(1)
                    natural_ranks.append(14)
                    natural_ranks.sort()

        gaps = 0
        for i in range(len(natural_ranks) - 1):
            gaps += natural_ranks[i+1] - natural_ranks[i] - 1
            
        return gaps <= wildcard_count

    def _discard_phase(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Determines the action for the discard phase."""
        hand = observation.get("hand", [])
        if not hand:
            return {"action_type": "DISCARD", "card": None}

        card_to_discard = random.choice(hand)
        return {"action_type": "DISCARD", "card": card_to_discard}
