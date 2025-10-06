"""Tests for the JAX-based game environment."""

import pytest
import jax.numpy as jnp

from buraco.jax_env.state import init_game_state, JaxGameState
from buraco.jax_env.actions import DrawAction
from buraco.jax_env.logic import step
from buraco.jax_env.converter import to_jax_state
from buraco.models import GameState


def test_init_game_state():
    """Test that the JAX game state can be initialized."""
    state = init_game_state()
    assert isinstance(state, JaxGameState)
    assert state.card_location.shape == (28, 108)  # L_LOCATIONS = 28


def test_to_jax_state():
    """Test that a standard GameState can be converted to a JaxGameState."""
    game_state = GameState()
    jax_state = to_jax_state(game_state)
    assert isinstance(jax_state, JaxGameState)


def test_step_function():
    """Test the JAX step function."""
    state = init_game_state()
    action = DrawAction(choice=jnp.array([0], dtype=jnp.uint8))
    new_state = step(state, action)
    assert isinstance(new_state, JaxGameState)
