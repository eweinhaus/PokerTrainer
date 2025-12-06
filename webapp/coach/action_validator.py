"""
Action Validation Module

Centralized validation layer that checks action appropriateness before execution.
Provides clear error messages for invalid actions.
"""

# logging removed



class ActionValidator:
    """
    Validates actions before execution to ensure they are contextually appropriate.
    """
    
    @staticmethod
    def validate_action(action_value, game_state, player_id=0, legal_actions=None):
        """
        Validate that an action is appropriate for the current game context.
        
        Args:
            action_value (int): Action value to validate
            game_state (dict): Game state dictionary
            player_id (int): Player ID (0 for human, 1 for opponent)
            legal_actions (list, optional): List of legal action values
        
        Returns:
            tuple: (is_valid, error_message)
                - is_valid (bool): True if action is valid
                - error_message (str): Error message if invalid, None if valid
        """
        # Check if action is in legal actions
        if legal_actions is not None:
            if action_value not in legal_actions:
                # Special case: Action 2 (RAISE_HALF_POT) can be used if action 3 (RAISE_POT) is legal
                # Action 2 gets mapped to action 3 in the backend processing
                if action_value == 2 and 3 in legal_actions:
                    pass  # Allow action 2 since it maps to legal action 3
                else:
                    return False, f'Action {action_value} is not in legal actions: {legal_actions}'
        
        # Basic validation: action value must be 0-4
        if action_value < 0 or action_value > 4:
            return False, f'Invalid action value: {action_value}. Must be between 0 and 4.'
        
        # Context-specific validation
        try:
            raw_obs = game_state.get('raw_obs', {}) if isinstance(game_state, dict) else {}
            if not raw_obs:
                raw_obs = game_state if isinstance(game_state, dict) else {}
            
            stage = raw_obs.get('stage', 0)
            if hasattr(stage, 'value'):
                stage = stage.value
            elif not isinstance(stage, int):
                stage = int(stage) if stage else 0
            
            raised = raw_obs.get('raised', [0, 0])
            player_raised = raised[player_id] if player_id < len(raised) else 0
            opponent_raised = raised[1 - player_id] if len(raised) > 1 - player_id else 0
            
            # Validate Check/Call (action 1)
            if action_value == 1:
                # Check if facing a bet
                epsilon = 0.01
                is_facing_bet = (opponent_raised - player_raised) > epsilon
                
                # If not facing a bet, action should be "Check"
                # If facing a bet, action should be "Call"
                # Both are valid, so no additional validation needed
                pass
            
            # Validate raise actions (2 and 3)
            if action_value in [2, 3]:
                # Raises are only valid if player has chips
                stakes = raw_obs.get('stakes', [100, 100])
                player_stack = stakes[player_id] if player_id < len(stakes) else 100
                
                if player_stack <= 0:
                    return False, f'Cannot raise: player has no chips remaining'
            
            # All validations passed
            return True, None
            
        except Exception as e:
            # Don't fail validation on error, just log it
            return True, None
    
    @staticmethod
    def validate_action_for_context(action_value, context):
        """
        Validate action using context dictionary from ActionLabeling.
        
        Args:
            action_value (int): Action value to validate
            context (dict): Context dictionary from ActionLabeling.get_context_from_state()
        
        Returns:
            tuple: (is_valid, error_message)
        """
        # Basic validation
        if action_value < 0 or action_value > 4:
            return False, f'Invalid action value: {action_value}'
        
        # Context-specific checks
        is_facing_bet = context.get('is_facing_bet', False)
        
        # Check/Call validation
        if action_value == 1:
            # Always valid (can check or call)
            return True, None
        
        # Raise validation
        if action_value in [2, 3]:
            # Raises are valid in most contexts
            # Additional checks could be added here
            return True, None
        
        # Fold and All-In are always valid if legal
        if action_value in [0, 4]:
            return True, None
        
        return True, None

