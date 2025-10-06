"""JAX-based game state representation for Buraco."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# Constants from MDP.md (can be configured)
M_MAX = 20  # Max number of melds
F_CARD = 20  # Number of static card features
F_MELD = 10  # Number of meld features
F_META = 32  # Number of turn/match meta features
L_LOCATIONS = 4 + M_MAX + 4  # Total number of locations


class JaxGameState(NamedTuple):
    """Represents the game state using JAX arrays for RL training."""

    # Card-centric tensors
    card_location: Array  # Shape: [L_LOCATIONS, 108], dtype=uint8
    stack_pos: Array  # Shape: [2, 108], dtype=float32
    card_static: Array  # Shape: [F_CARD, 108], dtype=float32

    # Meld-centric tensors
    meld_table: Array  # Shape: [M_MAX, F_MELD], dtype=float32
    meld_mask: Array  # Shape: [M_MAX], dtype=uint8

    # Game-level tensors
    turn_meta: Array  # Shape: [F_META], dtype=float32

    # Action masks (provided by the environment)
    mask_draw_actiontype: Array  # Shape: [2], dtype=uint8
    mask_can_meld_new: Array  # Shape: [1], dtype=uint8
    mask_meld_new_card: Array  # Shape: [11], dtype=uint8 (assuming max hand size)
    mask_can_meld_add: Array  # Shape: [M_MAX], dtype=uint8
    mask_meld_add_card: Array  # Shape: [M_MAX, 11], dtype=uint8
    mask_discard_card: Array  # Shape: [108], dtype=uint8


def init_game_state() -> JaxGameState:
    """Initializes a new game state with empty JAX arrays."""
    return JaxGameState(
        card_location=jnp.zeros((L_LOCATIONS, 108), dtype=jnp.uint8),
        stack_pos=jnp.zeros((2, 108), dtype=jnp.float32),
        card_static=jnp.zeros((F_CARD, 108), dtype=jnp.float32),
        meld_table=jnp.zeros((M_MAX, F_MELD), dtype=jnp.float32),
        meld_mask=jnp.zeros((M_MAX,), dtype=jnp.uint8),
        turn_meta=jnp.zeros((F_META,), dtype=jnp.float32),
        mask_draw_actiontype=jnp.zeros((2,), dtype=jnp.uint8),
        mask_can_meld_new=jnp.zeros((1,), dtype=jnp.uint8),
        mask_meld_new_card=jnp.zeros((11,), dtype=jnp.uint8),
        mask_can_meld_add=jnp.zeros((M_MAX,), dtype=jnp.uint8),
        mask_meld_add_card=jnp.zeros((M_MAX, 11), dtype=jnp.uint8),
        mask_discard_card=jnp.zeros((108,), dtype=jnp.uint8),
    )
