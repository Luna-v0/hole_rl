"""JAX-based action representation for Buraco."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# Constants
MAX_MELD_OPS = 5  # Max number of meld operations in a single turn


class DrawAction(NamedTuple):
    """Draw action: 0 for DRAW_STOCK, 1 for PICK_DISCARD."""
    choice: Array  # Shape: [1], dtype=uint8


class MeldAction(NamedTuple):
    """Meld action: a sequence of meld operations."""
    ops: Array  # Shape: [MAX_MELD_OPS, 3], dtype=int32
    # Each op: [op_type, card_index, meld_id]


class DiscardAction(NamedTuple):
    """Discard action: the card_id to discard."""
    card_id: Array  # Shape: [1], dtype=int32


class JaxActions(NamedTuple):
    """A container for all possible actions in a turn."""
    draw: DrawAction
    meld: MeldAction
    discard: DiscardAction
