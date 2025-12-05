#!/usr/bin/env python3
"""
Test script to verify that when facing a 4-bet, only fold, call, and all-in options are shown.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'webapp'))

from webapp.coach.action_labeling import ActionLabeling

def test_facing_4bet_options():
    """Test that facing a 4-bet shows only fold/call/all-in options"""

    # Simulate state where player is facing a 4-bet (opponent raised to 25BB = 50 chips)
    game_state = {
        'raw_obs': {
            'hand': ['DT', 'H9'],  # Player's hand
            'public_cards': [],
            'all_chips': [np.int64(8), np.int64(4)],
            'my_chips': np.int64(4),
            'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
            'stakes': [np.int64(92), np.int64(96)],  # Player has 92, AI has 96 (facing 50 chip bet)
            'current_player': 0,  # Player's turn
            'pot': np.int64(75),  # Pot after 4-bet
            'stage': Stage.PREFLOP,
            'raised': [0, 50]  # Player has 0, opponent has 50 (25BB)
        }
    }

    # Test the context detection
    context = ActionLabeling.get_context_from_state(game_state, player_id=0)

    # Test the button labels
    labels = ActionLabeling.get_button_labels(context)

    # Should hide raise buttons when facing 4-bet (betting_level = 2)
    expected_show_raise = False
    if labels['showRaiseHalfPot'] == expected_show_raise and labels['showRaisePot'] == expected_show_raise:
        return True
    else:
        return False

def test_facing_3bet_options():
    """Test that facing a 3-bet still shows 4-bet options"""

    # Simulate state where player is facing a 3-bet (opponent raised to 10BB = 20 chips)
    game_state = {
        'raw_obs': {
            'hand': ['DT', 'H9'],  # Player's hand
            'public_cards': [],
            'all_chips': [np.int64(8), np.int64(4)],
            'my_chips': np.int64(4),
            'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
            'stakes': [np.int64(92), np.int64(96)],  # Player has 92, AI has 96 (facing 20 chip bet)
            'current_player': 0,  # Player's turn
            'pot': np.int64(30),  # Pot after 3-bet
            'stage': Stage.PREFLOP,
            'raised': [0, 20]  # Player has 0, opponent has 20 (10BB)
        }
    }

    # Test the context detection
    context = ActionLabeling.get_context_from_state(game_state, player_id=0)

    # Test the button labels
    labels = ActionLabeling.get_button_labels(context)

    # Should show raise buttons when facing 3-bet (betting_level = 1)
    expected_show_raise = True
    expected_label = '4-bet to 25 BB'
    if (labels['showRaiseHalfPot'] == expected_show_raise and
        labels['showRaisePot'] == False and  # Only show one raise button
        labels['raiseHalfPot'] == expected_label):
        return True
    else:
        return False

if __name__ == "__main__":
    try:
        # Mock the required imports
        import numpy as np
        from enum import Enum

        class Action(Enum):
            FOLD = 0
            CHECK_CALL = 1
            RAISE_HALF_POT = 2
            RAISE_POT = 3
            ALL_IN = 4

        class Stage(Enum):
            PREFLOP = 0
            FLOP = 1
            TURN = 2
            RIVER = 3

        success1 = test_facing_4bet_options()
        success2 = test_facing_3bet_options()

        if success1 and success2:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
