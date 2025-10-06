"""JAX-based game logic for Buraco."""

import jax
import jax.numpy as jnp

from .state import JaxGameState
from .actions import DrawAction


@jax.jit
def draw_from_stock(state: JaxGameState) -> JaxGameState:
    """Player draws a card from the stock."""
    # This is a simplified example. A real implementation would need to handle
    # the card mapping and update the card_location matrix correctly.
    # For now, we just return the state as is.
    return state


@jax.jit
def take_discard_pile(state: JaxGameState) -> JaxGameState:
    """Player takes the discard pile."""
    # Similar to draw_from_stock, this is a placeholder.
    return state


@jax.jit
def step(state: JaxGameState, action: DrawAction) -> JaxGameState:
    """Performs a single step in the game (draw phase)."""

    def _draw_from_stock(_):
        return draw_from_stock(state)

    def _take_discard_pile(_):
        return take_discard_pile(state)

    # Use jax.lax.cond to choose the action based on action.choice
    new_state = jax.lax.cond(
        action.choice[0] == 0,
        _draw_from_stock,
        _take_discard_pile,
        operand=None,
    )

    return new_state
