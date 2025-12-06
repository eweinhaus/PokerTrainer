#!/usr/bin/env python3
"""
Test script to verify the Call/Check labeling fix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from webapp.coach.action_labeling import ActionLabeling

def test_call_check_labeling():
    """Test that Check/Call actions are labeled correctly based on context"""

    # Test case 1: Facing a bet (should be "Call")
    # Simulate a state where player faces an open raise
    mock_state = {
        'raw_obs': {
            'stage': 0,  # preflop
            'raised': [1.0, 3.0],  # Player has 1BB (SB), opponent has 3BB (open raise)
            'big_blind': 2,
            'pot': 4.5
        },
        'in_chips': [99.0, 97.0],  # Player has called 1BB, opponent has raised to 3BB
        'raised': [1.0, 3.0]
    }

    # Get context using the proper method
    context_facing_bet = ActionLabeling.get_context_from_state(mock_state, player_id=0, env=None)
    print(f"Test 1 extracted context: {context_facing_bet}")

    print(f"Test 1 context: {context_facing_bet}")
    label_facing_bet = ActionLabeling.get_action_label(1, context_facing_bet, bet_amount=0)
    print(f"Test 1 - Facing bet context: {label_facing_bet}")
    assert label_facing_bet == "Call", f"Expected 'Call', got '{label_facing_bet}'"

    # Test case 2: Not facing a bet (should be "Check")
    context_not_facing_bet = {
        'is_preflop': True,
        'is_small_blind': False,
        'is_first_to_act': True,
        'is_facing_bet': False,  # Key: not facing a bet
        'betting_level': 0,
        'big_blind': 2,
        'pot': 1.5,  # Only blinds
        'opponent_raised': 1.0,  # BB
        'player_raised': 0.5    # SB
    }

    label_not_facing_bet = ActionLabeling.get_action_label(1, context_not_facing_bet, bet_amount=0)
    print(f"Test 2 - Not facing bet context: {label_not_facing_bet}")
    assert label_not_facing_bet == "Check", f"Expected 'Check', got '{label_not_facing_bet}'"

    # Test case 3: SB facing BB blind (special case - should be "Call")
    context_sb_facing_bb = {
        'is_preflop': True,
        'is_small_blind': True,   # SB
        'is_first_to_act': False,
        'is_facing_bet': True,   # Facing BB's blind
        'betting_level': 0,
        'big_blind': 2,
        'pot': 1.5,
        'opponent_raised': 1.0,  # BB posted
        'player_raised': 0.0     # SB not yet acted
    }

    label_sb_facing_bb = ActionLabeling.get_action_label(1, context_sb_facing_bb, bet_amount=0)
    print(f"Test 3 - SB facing BB blind: {label_sb_facing_bb}")
    assert label_sb_facing_bb == "Call", f"Expected 'Call', got '{label_sb_facing_bb}'"

    # Test case 4: Postflop facing bet (should be "Call")
    context_postflop_facing_bet = {
        'is_preflop': False,
        'is_small_blind': False,
        'is_first_to_act': False,
        'is_facing_bet': True,   # Facing bet on flop
        'betting_level': 0,
        'big_blind': 2,
        'pot': 6.0,
        'opponent_raised': 2.0,  # Bet 1 BB
        'player_raised': 0.0
    }

    label_postflop_facing_bet = ActionLabeling.get_action_label(1, context_postflop_facing_bet, bet_amount=0)
    print(f"Test 4 - Postflop facing bet: {label_postflop_facing_bet}")
    assert label_postflop_facing_bet == "Call", f"Expected 'Call', got '{label_postflop_facing_bet}'"

    # Test case 5: Postflop not facing bet (should be "Check")
    context_postflop_not_facing_bet = {
        'is_preflop': False,
        'is_small_blind': False,
        'is_first_to_act': True,
        'is_facing_bet': False,  # First to act postflop
        'betting_level': 0,
        'big_blind': 2,
        'pot': 4.5,
        'opponent_raised': 0.0,
        'player_raised': 0.0
    }

    label_postflop_not_facing_bet = ActionLabeling.get_action_label(1, context_postflop_not_facing_bet, bet_amount=0)
    print(f"Test 5 - Postflop not facing bet: {label_postflop_not_facing_bet}")
    assert label_postflop_not_facing_bet == "Check", f"Expected 'Check', got '{label_postflop_not_facing_bet}'"

    print("\n✅ All action labeling tests passed!")
    return True

if __name__ == "__main__":
    test_call_check_labeling()
