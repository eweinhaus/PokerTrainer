#!/usr/bin/env python3
"""
Test script to verify the 4-bet labeling fix for AI opponent.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'webapp'))

from webapp.coach.action_labeling import ActionLabeling

def test_4bet_labeling():
    """Test that AI opponent correctly shows 4-bet when facing player 3-bet"""

    # Simulate state where AI (player 1) is facing a player 3-bet
    # Player raised to 10BB (20 chips), AI is facing this bet
    game_state = {
        'raw_obs': {
            'hand': ['DT', 'H9'],  # AI's hand
            'public_cards': [],
            'all_chips': [np.int64(8), np.int64(4)],
            'my_chips': np.int64(4),
            'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
            'stakes': [np.int64(92), np.int64(96)],  # Player has 92, AI has 96 (facing 20 chip bet)
            'current_player': 1,
            'pot': np.int64(12),
            'stage': Stage.PREFLOP,
            'raised': [20, 0]  # Player raised to 20, AI has 0
        }
    }

    # Test the context detection
    context = ActionLabeling.get_context_from_state(game_state, player_id=1)

    # Test the action labeling for RAISE_HALF_POT (action 2)
    action_label = ActionLabeling.get_action_label(2, context, bet_amount=30)  # AI raising to 30 total

    # Expected: "4-bet to 15BB" (since facing a 3-bet to 10BB)
    expected = "4-bet to 15BB"
    if action_label == expected:
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

        success = test_4bet_labeling()
        sys.exit(0 if success else 1)

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
