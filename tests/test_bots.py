"""Tests for the bot functionality using pytest."""

import pytest
from uuid import UUID

from buraco.game import GameManager
from buraco.models import Player, TurnPhase


@pytest.fixture
def game_manager() -> GameManager:
    """Pytest fixture for the GameManager."""
    return GameManager()


def test_add_bot_player(game_manager: GameManager):
    """Test that a bot can be added to a game."""
    game = game_manager.create_game()
    bot_player = game_manager.add_bot_player(game.game_id, "Test Bot")
    assert bot_player is not None
    assert bot_player.is_bot
    assert len(game.players) == 1


def test_start_game_with_one_bot(game_manager: GameManager):
    """Test starting a game with one human and one bot player."""
    game = game_manager.create_game()
    game_manager.join_game(game.game_id, "Human Player")
    game_manager.add_bot_player(game.game_id, "Test Bot")

    started_game = game_manager.start_game(game.game_id)
    assert started_game is not None
    assert started_game.game_started
    assert len(started_game.players[0].hand) == 11
    assert len(started_game.players[1].hand) == 11


def test_bot_draws_card(game_manager: GameManager):
    """Test that a bot draws a card."""
    game = game_manager.create_game()
    bot_player = game_manager.add_bot_player(game.game_id, "Test Bot")
    game_manager.join_game(game.game_id, "Human Player")
    game = game_manager.start_game(game.game_id)

    # It's bot's turn if it's the first player
    if game.players[0].is_bot:
        initial_hand_size = len(bot_player.hand)
        game = game_manager.play_bot_turn(game.game_id)
        # Hand size should be the same after drawing and discarding
        assert len(bot_player.hand) == initial_hand_size


def test_bot_discards_card(game_manager: GameManager):
    """Test that a bot discards a card."""
    game = game_manager.create_game()
    bot_player = game_manager.add_bot_player(
        game.game_id, "Test Bot", algorithm="deterministic"
    )
    human_player = game_manager.join_game(game.game_id, "Human Player")
    game = game_manager.start_game(game.game_id)

    bot_player_index = -1
    for i, p in enumerate(game.players):
        if p.is_bot:
            bot_player_index = i
            break

    game.current_turn_player_index = bot_player_index

    initial_discard_pile_size = len(game.discard_pile)

    # Manually play the bot's turn
    game_manager.draw_card_from_deck(game.game_id, bot_player.player_id)
    card_to_discard = bot_player.hand[0]
    game.turn_phase = TurnPhase.DISCARD
    game_manager.discard_card(game.game_id, bot_player.player_id, card_to_discard)

    assert len(game.discard_pile) == initial_discard_pile_size + 1

    def test_two_bots_play_a_turn(game_manager: GameManager):
        """Test that two bots can play a turn one after the other."""
        game = game_manager.create_game()
        bot1 = game_manager.add_bot_player(game.game_id, "Test Bot 1")
        bot2 = game_manager.add_bot_player(game.game_id, "Test Bot 2")
        game = game_manager.start_game(game.game_id)

        initial_turn = game.current_turn_player_index
        game_manager.play_bot_turn(game.game_id)
        game = game_manager.get_game(game.game_id)
        # After one bot plays, the turn should advance
        assert game.current_turn_player_index != initial_turn
def test_four_bots_play_a_turn(game_manager: GameManager):
    """Test that four bots can play a turn one after the other."""
    game = game_manager.create_game()
    bot1 = game_manager.add_bot_player(game.game_id, "Test Bot 1")
    bot2 = game_manager.add_bot_player(game.game_id, "Test Bot 2")
    bot3 = game_manager.add_bot_player(game.game_id, "Test Bot 3")
    bot4 = game_manager.add_bot_player(game.game_id, "Test Bot 4")
    game = game_manager.start_game(game.game_id)

    initial_turn = game.current_turn_player_index
    game_manager.play_bot_turn(game.game_id)
    game = game_manager.get_game(game.game_id)
    # After one bot plays, the turn should advance
    assert game.current_turn_player_index != initial_turn