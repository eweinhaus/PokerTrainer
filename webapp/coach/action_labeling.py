"""
Shared Action Labeling Module

Single source of truth for action labeling logic used by frontend, backend, and LLM opponent.
Consolidates context detection and label generation to ensure 100% consistency.
"""

# logging removed



class ActionLabeling:
    """
    Centralized action labeling logic for consistent behavior across frontend, backend, and LLM opponent.
    
    This module provides:
    - Context extraction from game state
    - Button label generation for UI
    - Action label generation for history display
    """
    
    @staticmethod
    def get_context_from_state(game_state, player_id=0, env=None):
        """
        Extract context information from game state.
        
        Args:
            game_state (dict): Game state dictionary with 'raw_obs' key
            player_id (int): Player ID (0 for human, 1 for opponent)
            env: Optional RLCard environment for additional context
        
        Returns:
            dict: Context dictionary with keys:
                - is_preflop (bool): True if preflop stage
                - is_small_blind (bool): True if player is small blind
                - is_first_to_act (bool): True if player is first to act
                - is_facing_bet (bool): True if player is facing a bet
                - betting_level (int): 0=facing open, 1=facing 3-bet, 2+=facing 4-bet+
                - big_blind (float): Big blind amount
                - pot (float): Current pot size
                - opponent_raised (float): Opponent's current bet
                - player_raised (float): Player's current bet
        """
        raw_obs = game_state.get('raw_obs', {}) if isinstance(game_state, dict) else {}
        if not raw_obs:
            # Fallback: assume game_state is raw_obs
            raw_obs = game_state if isinstance(game_state, dict) else {}
        
        # Extract stage
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        elif not isinstance(stage, int):
            stage = int(stage) if stage else 0
        
        is_preflop = stage == 0
        
        # Extract raised amounts
        # Handle case where raised is None (can happen after AI actions)
        raised = raw_obs.get('raised')
        if raised is None:
            raised = [0, 0]
        elif not isinstance(raised, (list, tuple)) or len(raised) < 2:
            raised = [0, 0]
        
        player_raised = raised[player_id] if player_id < len(raised) else 0
        opponent_raised = raised[1 - player_id] if len(raised) > 1 - player_id else 0
        
        # Extract other game info
        big_blind = raw_obs.get('big_blind', 2)
        pot = raw_obs.get('pot', 0)
        
        # CRITICAL FIX: If raised array shows [0, 0] but pot indicates a raise occurred,
        # infer the correct raised amounts from pot and in_chips
        # This handles cases where the environment doesn't update raised correctly after actions
        if pot > 0:
            # Try to get in_chips from game_state (more reliable than raised array)
            in_chips = game_state.get('in_chips', [0, 0]) if isinstance(game_state, dict) else [0, 0]
            
            # Fallback: Get in_chips from environment if not in game_state
            if (not in_chips or len(in_chips) < 2) and env is not None:
                try:
                    if hasattr(env, 'game') and hasattr(env.game, 'players'):
                        in_chips = [int(p.in_chips) for p in env.game.players]
                except Exception:
                    pass
            
            if in_chips and len(in_chips) >= 2:
                # in_chips represents total chips each player has put in this hand
                # For current betting round, we need to compare relative amounts
                # If opponent has put in more than player, player is facing a bet
                player_in_chips = in_chips[player_id] if player_id < len(in_chips) else 0
                opponent_in_chips = in_chips[1 - player_id] if len(in_chips) > 1 - player_id else 0
                
                # If raised array is wrong but in_chips shows a difference, use in_chips
                if (player_raised == 0 and opponent_raised == 0 and 
                    abs(opponent_in_chips - player_in_chips) > 0.01):
                    # Use in_chips difference as the raised amounts for current betting round
                    # This is an approximation - in_chips is cumulative, but for preflop
                    # it should be close enough for determining if facing a bet
                    if opponent_in_chips > player_in_chips:
                        opponent_raised = opponent_in_chips - player_in_chips
                    elif player_in_chips > opponent_in_chips:
                        player_raised = player_in_chips - opponent_in_chips
                
                # Alternative: Infer from pot size for preflop
                # After button open: pot = SB (0.5BB) + BB (1BB) + raise (6 chips) = 9 chips = 4.5BB
                # After 3-bet: pot = SB + BB + open (6) + 3bet (18) = 27 chips = 13.5BB
                if is_preflop and player_raised == 0 and opponent_raised == 0:
                    small_blind = big_blind // 2
                    blinds_total = big_blind + small_blind
                    pot_bb = pot / big_blind if big_blind > 0 else 0
                    
                    # If pot is larger than just blinds, someone has raised
                    if pot > blinds_total:
                        # Infer opponent_raised from pot
                        # pot = blinds_total + opponent_raise (if opponent raised)
                        # OR pot = blinds_total + player_raise (if player raised)
                        # We need to check who acted last - if it's player's turn, opponent likely raised
                        # For now, assume if pot > blinds, opponent raised (common case)
                        inferred_opponent_raise = pot - blinds_total
                        if inferred_opponent_raise > 0:
                            opponent_raised = inferred_opponent_raise
        
        # Determine position (small blind)
        is_small_blind = False
        dealer_id = None
        # Check for dealer_id in env.game (mock) or directly on env (real RLCard)
        if env is not None:
            if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                dealer_id = env.game.dealer_id
            elif hasattr(env, 'dealer_id'):
                dealer_id = env.dealer_id

            if dealer_id is not None:
                button_id = (dealer_id + 1) % 2
                is_small_blind = button_id == player_id
        
        # Determine if facing a bet
        epsilon = 0.01
        if not is_preflop:
            # Postflop: if raised amounts are equal, it's a check
            if abs(opponent_raised - player_raised) <= epsilon:
                is_facing_bet = False
            else:
                is_facing_bet = (opponent_raised - player_raised) > epsilon
        else:
            # Preflop: If opponent has raised more than player, player is facing a bet
            # This includes the case where SB (1 BB) faces BB (2 BB) - SB must call BB's bet
            # Special case: SB always faces BB's bet preflop (unless BB has folded, which doesn't happen)
            if is_small_blind and opponent_raised >= big_blind * 0.9 and player_raised < big_blind * 0.9:
                # SB facing BB's bet (blind post scenario)
                is_facing_bet = True
            else:
                # Standard logic: facing bet if opponent raised more than player
                is_facing_bet = (opponent_raised - player_raised) > epsilon
        
        # Determine if first to act
        if is_preflop:
            # First to act preflop if not facing a bet (no raises have occurred yet)
            is_first_to_act = not is_facing_bet
        else:
            is_first_to_act = player_raised == opponent_raised
        
        # Determine betting level (for preflop facing bet scenarios)
        betting_level = 0
        if is_preflop and is_facing_bet:
            opponent_raised_bb = opponent_raised / big_blind if big_blind > 0 else 0
            if opponent_raised_bb <= 4.0:
                betting_level = 0  # Facing an open
            elif opponent_raised_bb <= 12.0:
                betting_level = 1  # Facing a 3-bet
            else:
                betting_level = 2  # Facing 4-bet or higher
        
        # Convert all values to Python native types to prevent numpy boolean evaluation errors
        return {
            'is_preflop': bool(is_preflop),
            'is_small_blind': bool(is_small_blind),
            'is_first_to_act': bool(is_first_to_act),
            'is_facing_bet': bool(is_facing_bet),
            'betting_level': int(betting_level),
            'big_blind': float(big_blind),
            'pot': float(pot),
            'opponent_raised': float(opponent_raised),
            'player_raised': float(player_raised)
        }
    
    @staticmethod
    def get_button_labels(context):
        """
        Return button labels for given context (matches frontend logic).
        
        Args:
            context (dict): Context dictionary from get_context_from_state()
        
        Returns:
            dict: Button labels with keys:
                - raiseHalfPot (str): Label for RAISE_HALF_POT button
                - raisePot (str): Label for RAISE_POT button
                - showRaiseHalfPot (bool): Whether to show RAISE_HALF_POT button
                - checkCall (str): Label for CHECK_CALL button ("Check" or "Call")
        """
        is_preflop = context.get('is_preflop', False)
        is_small_blind = context.get('is_small_blind', False)
        is_first_to_act = context.get('is_first_to_act', False)
        is_facing_bet = context.get('is_facing_bet', False)
        betting_level = context.get('betting_level', 0)
        big_blind = context.get('big_blind', 2)
        opponent_raised = context.get('opponent_raised', 0)
        player_raised = context.get('player_raised', 0)
        
        labels = {
            'raiseHalfPot': 'Raise ½ Pot',
            'raisePot': 'Raise Pot',
            'showRaiseHalfPot': True,
            'showRaisePot': True,
            'checkCall': 'Check' if not is_facing_bet else 'Call'
        }
        
        if is_preflop:
            # PREFLOP LOGIC - All preflop decisions should be in BB
            # IMPORTANT: Check is_facing_bet FIRST
            if is_facing_bet:
                # Facing a bet (big blind vs open, or facing 3-bet/4-bet)
                # Special case: SB facing BB's blind post (2 BB vs 1 BB) should show "Raise to 3 BB"
                if is_small_blind and opponent_raised <= big_blind * 1.1 and player_raised < big_blind * 0.9:
                    # SB facing BB's blind post - this is an opening scenario, not a 3-bet
                    labels['raiseHalfPot'] = 'Raise to 3 BB'
                    labels['raisePot'] = 'Raise Pot'
                    labels['showRaisePot'] = False
                elif betting_level == 0:
                    # Facing an open (2.5-3BB), show 3-bet option
                    labels['raiseHalfPot'] = '3-bet to 10 BB'
                    labels['raisePot'] = '3-bet to 10 BB'
                    labels['showRaisePot'] = False
                elif betting_level == 1:
                    # Facing a 3-bet, show 4-bet option
                    labels['raiseHalfPot'] = '4-bet to 25 BB'
                    labels['raisePot'] = '4-bet to 25 BB'
                    labels['showRaisePot'] = False
                else:
                    # Facing 4-bet or higher, only show fold/call/all-in (no 5-betting)
                    labels['raiseHalfPot'] = 'All-In'
                    labels['raisePot'] = 'All-In'
                    labels['showRaiseHalfPot'] = False
                    labels['showRaisePot'] = False
            elif is_small_blind and is_first_to_act:
                # Small blind opening (unopened pot, only blinds posted)
                labels['raiseHalfPot'] = 'Raise to 3 BB'
                labels['raisePot'] = 'Raise Pot'
                labels['showRaisePot'] = False
            else:
                # Big blind not facing a bet (can still raise)
                labels['raiseHalfPot'] = 'Raise to 3 BB'
                labels['raisePot'] = 'Raise Pot'
                labels['showRaisePot'] = False
        else:
            # POSTFLOP LOGIC - Always show TWO bet size options
            if is_first_to_act:
                # First to act (continuation bet, value bet, etc.)
                labels['raiseHalfPot'] = 'Bet ½ Pot'
                labels['raisePot'] = 'Bet ⅔ Pot'
            elif is_facing_bet:
                # Facing a bet (raise sizing based on opponent's bet)
                bet_to_call = opponent_raised - player_raised
                raise_25x_total = bet_to_call * 2.5 + bet_to_call
                raise_3x_total = bet_to_call * 3 + bet_to_call
                raise_25x_bb = round(raise_25x_total / big_blind)
                raise_3x_bb = round(raise_3x_total / big_blind)
                labels['raiseHalfPot'] = f'Raise to {raise_25x_bb} BB'
                labels['raisePot'] = f'Raise to {raise_3x_bb} BB'
            else:
                # Other postflop situations
                labels['raiseHalfPot'] = 'Bet ½ Pot'
                labels['raisePot'] = 'Bet ⅔ Pot'
        
        return labels
    
    @staticmethod
    def get_action_label(action_value, context, bet_amount=None):
        """
        Return action label for action history (matches backend _action_to_string logic).
        
        Args:
            action_value (int): Action value (0=Fold, 1=Check/Call, 2=Raise ½ Pot, 3=Raise Pot, 4=All-In)
            context (dict): Context dictionary from get_context_from_state()
            bet_amount (float, optional): Actual bet amount for more accurate labeling
        
        Returns:
            str: Action label string
        """
        if action_value == 0:
            return 'Fold'
        elif action_value == 1:
            # Check or Call - always use context-based detection
            # bet_amount is not reliable for CHECK_CALL actions (always 0 in tracking)
            is_facing_bet = context.get('is_facing_bet', False)
            if not is_facing_bet:
                # Not facing a bet, so it must be a Check
                return 'Check'
            else:
                # Facing a bet, so it must be a Call
                return 'Call'
        elif action_value == 2:
            # RAISE_HALF_POT
            return ActionLabeling._get_raise_label(context, bet_amount, is_half_pot=True)
        elif action_value == 3:
            # RAISE_POT
            return ActionLabeling._get_raise_label(context, bet_amount, is_half_pot=False)
        elif action_value == 4:
            return 'All-In'
        else:
            return f'Action {action_value}'
    
    @staticmethod
    def _get_raise_label(context, bet_amount=None, is_half_pot=True):
        """Get label for raise actions (action 2 or 3)"""
        is_preflop = context.get('is_preflop', False)
        is_small_blind = context.get('is_small_blind', False)
        is_first_to_act = context.get('is_first_to_act', False)
        is_facing_bet = context.get('is_facing_bet', False)
        betting_level = context.get('betting_level', 0)
        big_blind = context.get('big_blind', 2)
        opponent_raised = context.get('opponent_raised', 0)
        player_raised = context.get('player_raised', 0)
        
        if is_preflop:
            # PRIORITY 1: Use bet_amount if available - it's the most reliable indicator
            if bet_amount is not None and bet_amount > 0:
                bet_amount_bb = bet_amount / big_blind if big_blind > 0 else 0
                # bet_amount represents the total amount raised TO (not additional)
                if 2.5 <= bet_amount_bb <= 3.5:
                    # Open raise to 3BB
                    return 'Raise to 3BB'
                elif 9.5 <= bet_amount_bb <= 10.5:
                    # 3-bet to 10BB
                    return '3-bet to 10BB'
                elif 24.5 <= bet_amount_bb <= 25.5:
                    # 4-bet to 25BB
                    return '4-bet to 25BB'
                elif 17.5 <= bet_amount_bb <= 18.5:
                    # 4-bet to 18BB
                    return '4-bet to 18BB'
                elif 6.5 <= bet_amount_bb <= 8.5:
                    # 3-bet to 7-8BB OR 4-bet to 7-8BB (depending on betting_level)
                    # If betting_level == 1, this is a 4-bet (facing a 3-bet)
                    # If betting_level == 0, this is a 3-bet (facing an open)
                    if betting_level == 1:
                        return f'4-bet to {int(round(bet_amount_bb))}BB'
                    else:
                        return f'3-bet to {int(round(bet_amount_bb))}BB'
            
            # PRIORITY 2: Check if this is an open raise (SB facing BB's blind) vs 3-bet (facing an actual raise)
            # If opponent_raised is just the big blind (<= 1.1 * BB), this is an open, not a 3-bet
            is_facing_just_blind = (opponent_raised <= big_blind * 1.1 and 
                                   is_small_blind and 
                                   player_raised < big_blind * 0.9)
            
            if is_facing_bet and not is_facing_just_blind:
                # Facing an actual bet (not just the blind) - this is a 3-bet or 4-bet
                # Fallback: determine from betting level
                if betting_level == 0:
                    return '3-bet to 10BB'
                elif betting_level == 1:
                    return '4-bet to 25BB'
                else:
                    bet_to_call = opponent_raised - player_raised
                    raise_25x_total = bet_to_call * 2.5 + bet_to_call
                    raise_25x_bb = round(raise_25x_total / big_blind) if big_blind > 0 else 0
                    return f'Raise to {raise_25x_bb}BB'
            elif is_facing_bet and is_facing_just_blind:
                # SB facing BB's blind - this is an open raise, not a 3-bet
                return 'Raise to 3BB'
            elif is_small_blind and is_first_to_act:
                return 'Raise to 3BB'
            else:
                return 'Raise to 3BB'
        else:
            # Postflop
            if is_first_to_act:
                return 'Bet ½ Pot' if is_half_pot else 'Bet ⅔ Pot'
            elif is_facing_bet:
                bet_to_call = opponent_raised - player_raised
                if is_half_pot:
                    raise_25x_total = bet_to_call * 2.5 + bet_to_call
                    raise_25x_bb = round(raise_25x_total / big_blind) if big_blind > 0 else 0
                    return f'Raise to {raise_25x_bb}BB'
                else:
                    raise_3x_total = bet_to_call * 3 + bet_to_call
                    raise_3x_bb = round(raise_3x_total / big_blind) if big_blind > 0 else 0
                    return f'Raise to {raise_3x_bb}BB'
            else:
                return 'Bet ½ Pot' if is_half_pot else 'Bet ⅔ Pot'

