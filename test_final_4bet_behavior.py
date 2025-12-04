#!/usr/bin/env python3
"""
Final test to verify the complete 4-bet behavior change.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'webapp'))

from webapp.coach.action_labeling import ActionLabeling

def test_complete_4bet_workflow():
    """Test the complete workflow: 3-bet shows 4-bet option, 4-bet+ shows only fold/call/all-in"""

    print("Testing complete 4-bet workflow...")

    # Test 1: Facing a 3-bet (10BB) should show 4-bet option
    game_state_3bet = {
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

    context_3bet = ActionLabeling.get_context_from_state(game_state_3bet, player_id=0)
    labels_3bet = ActionLabeling.get_button_labels(context_3bet)

    print(f"Facing 3-bet (10BB): betting_level={context_3bet['betting_level']}")
    print(f"  Show raise buttons: {labels_3bet['showRaiseHalfPot']}, {labels_3bet['showRaisePot']}")
    print(f"  Raise label: {labels_3bet['raiseHalfPot']}")

    # Test 2: Facing a 4-bet (25BB+) should hide raise options
    game_state_4bet = {
        'raw_obs': {
            'hand': ['DT', 'H9'],
            'public_cards': [],
            'all_chips': [np.int64(8), np.int64(4)],
            'my_chips': np.int64(4),
            'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN],
            'stakes': [np.int64(92), np.int64(96)],
            'current_player': 0,
            'pot': np.int64(75),
            'stage': Stage.PREFLOP,
            'raised': [0, 50]  # Facing 25BB bet
        }
    }

    context_4bet = ActionLabeling.get_context_from_state(game_state_4bet, player_id=0)
    labels_4bet = ActionLabeling.get_button_labels(context_4bet)

    print(f"Facing 4-bet (25BB): betting_level={context_4bet['betting_level']}")
    print(f"  Show raise buttons: {labels_4bet['showRaiseHalfPot']}, {labels_4bet['showRaisePot']}")
    print(f"  Available actions: Fold, {labels_4bet['checkCall']}, All-In")

    # Verify expectations
    success = True

    # Facing 3-bet: should show 4-bet option
    if not (context_3bet['betting_level'] == 1 and
            labels_3bet['showRaiseHalfPot'] == True and
            labels_3bet['raiseHalfPot'] == '4-bet to 25 BB'):
        print("❌ FAILED: Facing 3-bet should show 4-bet option")
        success = False
    else:
        print("✅ Facing 3-bet correctly shows 4-bet option")

    # Facing 4-bet: should hide raise options
    if not (context_4bet['betting_level'] == 2 and
            labels_4bet['showRaiseHalfPot'] == False and
            labels_4bet['showRaisePot'] == False):
        print("❌ FAILED: Facing 4-bet should hide raise options")
        success = False
    else:
        print("✅ Facing 4-bet correctly hides raise options (only fold/call/all-in)")

    return success

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

        if test_complete_4bet_workflow():
            print("\n🎉 Complete 4-bet workflow test passed!")
            sys.exit(0)
        else:
            print("\n❌ Test failed")
            sys.exit(1)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
