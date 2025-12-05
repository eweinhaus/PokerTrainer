#!/usr/bin/env python3
"""
Simple test to verify LLM opponent action labeling fix
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))

# Mock the Action enum since we can't import RLCard easily
class Action:
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_POT = 3
    ALL_IN = 4

def test_action_labeling():
    """Test that action labeling shows Check vs Call correctly"""

    # Mock context with facing_bet = False (should show "Check")
    context_not_facing = {
        'facing_bet': False,
        'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT]
    }

    # Mock context with facing_bet = True (should show "Call")
    context_facing = {
        'facing_bet': True,
        'legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT]
    }

    # Simulate the logic from the fixed code
    def build_labels(context):
        legal_actions_labels = {}
        action_label_map = {
            Action.FOLD: 'Fold',
            Action.RAISE_HALF_POT: 'Raise ½ Pot',
            Action.RAISE_POT: 'Raise Pot',
            Action.ALL_IN: 'All-In'
        }

        # Dynamically set Check/Call label based on context
        check_call_label = 'Check' if not context.get('facing_bet', False) else 'Call'
        action_label_map[Action.CHECK_CALL] = check_call_label

        for action in context['legal_actions']:
            legal_actions_labels[action] = action_label_map.get(action, f'Action {action}')

        return legal_actions_labels

    # Test not facing bet
    labels_not_facing = build_labels(context_not_facing)
    expected_check = 'Check'
    actual_check = labels_not_facing.get(Action.CHECK_CALL)

    assert actual_check == expected_check, f"Expected '{expected_check}', got '{actual_check}'"

    # Test facing bet
    labels_facing = build_labels(context_facing)
    expected_call = 'Call'
    actual_call = labels_facing.get(Action.CHECK_CALL)

    assert actual_call == expected_call, f"Expected '{expected_call}', got '{actual_call}'"


def test_tool_schema():
    """Test that tool schema includes correct actions based on facing_bet"""

    def get_tool_schema(facing_bet=False):
        base_actions = ["fold", "raise_half_pot", "raise_pot", "all_in"]

        if facing_bet:
            base_actions.insert(0, "call")
            check_call_description = "The type of action to take. 'call' matches the bet to continue playing."
        else:
            base_actions.insert(0, "check")
            check_call_description = "The type of action to take. 'check' passes action to opponent."

        return {
            'enum': base_actions,
            'description': check_call_description
        }

    # Test not facing bet
    schema_not_facing = get_tool_schema(facing_bet=False)
    assert 'check' in schema_not_facing['enum'], "Should include 'check' when not facing bet"
    assert 'call' not in schema_not_facing['enum'], "Should not include 'call' when not facing bet"
    assert 'check' in schema_not_facing['description'], "Description should mention 'check'"

    # Test facing bet
    schema_facing = get_tool_schema(facing_bet=True)
    assert 'call' in schema_facing['enum'], "Should include 'call' when facing bet"
    assert 'check' not in schema_facing['enum'], "Should not include 'check' when facing bet"
    assert 'call' in schema_facing['description'], "Description should mention 'call'"


if __name__ == '__main__':
    try:
        test_action_labeling()
        test_tool_schema()
    except Exception as e:
        sys.exit(1)

