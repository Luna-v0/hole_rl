"""Factory for creating bot players."""

from typing import Dict, Any

from buraco.models import Player
from .random_bot import RandomBot
from .deterministic_bot import DeterministicBot


class BotFactory:
    """Factory for creating bot players."""

    def __init__(self):
        """Initializes the BotFactory."""
        self._bots = {
            "random": RandomBot,
            "deterministic": DeterministicBot,
        }

    def create_bot(self, algorithm: str, player: Player) -> Any:
        """Creates a bot player.

        Args:
            algorithm: The algorithm to use for the bot.
            player: The player instance for the bot.

        Returns:
            A bot instance.
        """
        bot_class = self._bots.get(algorithm)
        if not bot_class:
            raise ValueError(f"Unknown bot algorithm: {algorithm}")
        return bot_class(player)