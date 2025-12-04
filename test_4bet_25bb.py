#!/usr/bin/env python3
"""
Test script to verify the 4-bet 25BB functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'webapp'))

from webapp.coach.action_labeling import ActionLabeling

def test_4bet_25bb_labeling():
    """Test that 4-bet shows 25BB labels"""

    # Simulate state where player is facing a 3-bet (opponent raised to 20BB = 40 chips)
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
    print(f"Context for player facing 3-bet: {context}")

    # Test the button labels
    labels = ActionLabeling.get_button_labels(context)
    print(f"Button labels: {labels}")

    # Should show "4-bet to 25 BB" for both RAISE_HALF_POT and RAISE_POT
    expected = '4-bet to 25 BB'
    if labels['raiseHalfPot'] == expected and labels['raisePot'] == expected:
        print(f"✅ SUCCESS: Button labels show '{expected}'")
        return True
    else:
        print(f"❌ FAILED: Expected '{expected}', got raiseHalfPot='{labels['raiseHalfPot']}', raisePot='{labels['raisePot']}'")
        return False

def test_4bet_action_labeling():
    """Test that 4-bet actions are labeled correctly"""

    game_state = {
        'raw_obs': {
            'hand': ['DT', 'H9'],
            'public_cards': [],
            'all_chips': [np.int64(8), np.int64(4)],
            'my_chips': np.int64(4),
            'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
            'stakes': [np.int64(92), np.int64(96)],
            'current_player': 0,
            'pot': np.int64(30),
            'stage': Stage.PREFLOP,
            'raised': [0, 20]  # Facing 10BB bet
        }
    }

    context = ActionLabeling.get_context_from_state(game_state, player_id=0)

    # Test RAISE_POT action (which should be the 4-bet)
    action_label = ActionLabeling.get_action_label(3, context, bet_amount=50)  # 25BB total = 50 chips
    print(f"Action label for RAISE_POT (4-bet): {action_label}")

    expected = "4-bet to 25BB"
    if action_label == expected:
        print(f"✅ SUCCESS: Action labeled as '{expected}'")
        return True
    else:
        print(f"❌ FAILED: Expected '{expected}', got '{action_label}'")
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

        success1 = test_4bet_25bb_labeling()
        success2 = test_4bet_action_labeling()

        if success1 and success2:
            print("\n🎉 All 4-bet 25BB tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
