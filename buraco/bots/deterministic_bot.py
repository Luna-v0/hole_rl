"""A deterministic bot for testing."""

from typing import Dict, Any

from buraco.models import Player


class DeterministicBot:
    """A bot that plays deterministic actions for testing."""

    def __init__(self, player: Player):
        """Initializes the DeterministicBot."""
        self.player = player

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Gets the action from the bot based on the current game phase."""
        phase = observation.get("turn_phase")

        if phase == "draw":
            return self._draw_phase(observation)
        elif phase == "meld":
            return self._meld_phase(observation)
        elif phase == "discard":
            return self._discard_phase(observation)
        else:
            return {}

    def _draw_phase(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Determines the action for the draw phase."""
        return {"action_type": "DRAW", "choice": "DRAW_STOCK"}

    def _meld_phase(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Determines the action for the meld phase."""
        return {"action_type": "MELD", "ops": []}

    def _discard_phase(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Determines the action for the discard phase."""
        hand = observation.get("hand", [])
        if not hand:
            return {"action_type": "DISCARD", "card": None}

        card_to_discard = hand[0]
        return {"action_type": "DISCARD", "card": card_to_discard}
