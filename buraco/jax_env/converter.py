"""Converts the standard game state to the JAX-based representation."""

import jax.numpy as jnp

from buraco.models import GameState
from .state import JaxGameState, init_game_state


def to_jax_state(game_state: GameState) -> JaxGameState:
    """Converts a standard GameState to a JaxGameState.

    Args:
        game_state: The standard game state.

    Returns:
        The JAX-based game state.
    """
    jax_state = init_game_state()

    # This is a placeholder. A real implementation would need to map all the
    # fields from game_state to the JAX arrays in jax_state.
    # For example:
    # jax_state = jax_state._replace(
    #     turn_meta=jax_state.turn_meta.at[0].set(game_state.current_turn_player_index)
    # )

    return jax_state
