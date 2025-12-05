#!/usr/bin/env python3
"""
Test script to verify the legal actions fix
"""

try:
    import rlcard
    env = rlcard.make('no-limit-holdem')
    state, player_id = env.reset()

    # Simulate the fixed logic
    raw_obs = state['raw_obs']

    # Old logic (would fail)
    old_legal_actions = raw_obs.get('legal_actions', [])
    old_raw_legal_actions = raw_obs.get('raw_legal_actions', [])

    # New logic (should work)
    legal_actions = state.get('legal_actions', raw_obs.get('legal_actions', []))
    raw_legal_actions = state.get('raw_legal_actions', raw_obs.get('raw_legal_actions', []))

    # Test convert_actions function
    def convert_actions(actions):
        if not actions:
            return []
        result = []
        # Handle dict/OrderedDict (legal_actions is often a dict)
        if isinstance(actions, dict):
            actions = list(actions.keys())
        for action in actions:
            if hasattr(action, 'value'):
                # Action enum - use .value
                result.append(action.value)
            elif isinstance(action, int):
                result.append(action)
            else:
                # Try to convert to int
                try:
                    result.append(int(action))
                except (ValueError, TypeError):
                    # If all else fails, try to get the value attribute
                    action_value = getattr(action, 'action', None)
                    if action_value is not None:
                        result.append(action_value)
        return result

    converted_legal = convert_actions(legal_actions)
    converted_raw = convert_actions(raw_legal_actions)

    if converted_raw:
    else:

except ImportError:
