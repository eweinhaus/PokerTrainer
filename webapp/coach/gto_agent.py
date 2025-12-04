"""
GTO-based Agent for No Limit Hold'em

This agent uses GTO ranges and principles to make decisions instead of random actions.
"""

import random
import numpy as np
from .gto_rules import GTORules

# Import RLCard Action enum
try:
    from rlcard.games.nolimitholdem.round import Action
except ImportError:
    # Fallback if import fails
    class Action:
        FOLD = 0
        CHECK_CALL = 1
        RAISE_HALF_POT = 2
        RAISE_POT = 3
        ALL_IN = 4


class GTOAgent:
    """
    GTO-based agent that uses optimal ranges and strategy.
    
    This agent:
    - Uses GTO preflop ranges based on position and stack depth
    - Never open-shoves all-in at 50 BB (uses proper opening ranges)
    - Uses simple postflop strategy (bet strong, check/call medium, fold weak)
    """
    
    def __init__(self, num_actions):
        """
        Initialize GTO agent.
        
        Args:
            num_actions (int): Number of available actions
        """
        self.use_raw = True  # Use raw observations
        self.num_actions = num_actions
        self.gto_rules = GTORules()
        self.player_id = None  # Track which player we are
    
    def _card_to_rank_suit(self, card_index):
        """
        Convert RLCard card index to rank and suit.
        
        RLCard uses 0-51: suit = card // 13, rank = card % 13
        Rank: 0=A, 1=2, 2=3, ..., 12=K
        
        Args:
            card_index (int): RLCard card index (0-51)
        
        Returns:
            tuple: (rank, suit) where rank is 0-12 (A=0, K=12) and suit is 0-3
        """
        # Convert to int if it's a string or numpy type
        if not isinstance(card_index, int):
            try:
                card_index = int(card_index)
            except (ValueError, TypeError):
                # If conversion fails, default to 0 (shouldn't happen, but safe fallback)
                card_index = 0
        
        suit = card_index // 13
        rank = card_index % 13
        return (rank, suit)
    
    def _hand_to_string(self, hand):
        """
        Convert RLCard hand (list of card indices) to hand string format.
        
        Args:
            hand (list): List of 2 card indices
        
        Returns:
            str: Hand string like "AA", "AKs", "AKo", "72o"
        """
        if not hand or len(hand) != 2:
            return None
        
        rank_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        
        card1_rank, card1_suit = self._card_to_rank_suit(hand[0])
        card2_rank, card2_suit = self._card_to_rank_suit(hand[1])
        
        rank1_name = rank_names[card1_rank]
        rank2_name = rank_names[card2_rank]
        
        # Ensure rank1 >= rank2 (higher rank first)
        if card1_rank < card2_rank:
            rank1_name, rank2_name = rank2_name, rank1_name
            card1_rank, card2_rank = card2_rank, card1_rank
            card1_suit, card2_suit = card2_suit, card1_suit
        
        # Check if pair
        if card1_rank == card2_rank:
            return f"{rank1_name}{rank2_name}"
        
        # Check if suited
        if card1_suit == card2_suit:
            return f"{rank1_name}{rank2_name}s"
        else:
            return f"{rank1_name}{rank2_name}o"
    
    def _get_position(self, raw_obs):
        """
        Determine position from raw observation.
        
        In RLCard heads-up:
        - Player 0 is button (posts small blind, acts first preflop)
        - Player 1 is big blind (posts big blind, acts second preflop)
        
        Since this agent is used as the opponent (second agent in env.set_agents),
        we are always player 1 (big blind).
        
        Args:
            raw_obs (dict): Raw observation from RLCard
        
        Returns:
            str: "button" or "big_blind"
        """
        # We are always player 1 (big blind) since we're the opponent agent
        # Player 0 is the human player (button)
        return 'big_blind'
    
    def _get_stack_depth(self, raw_obs):
        """
        Get stack depth in big blinds.
        
        Args:
            raw_obs (dict): Raw observation
        
        Returns:
            float: Stack depth in big blinds
        """
        all_chips = raw_obs.get('all_chips', [0, 0])
        big_blind = raw_obs.get('big_blind', 2)
        
        if big_blind == 0:
            return 100.0  # Default
        
        # Get opponent's stack (player 1)
        if len(all_chips) > 1:
            stack_bb = all_chips[1] / big_blind
            return stack_bb
        
        return 100.0  # Default
    
    def _get_preflop_action_type(self, raw_obs):
        """
        Determine preflop action type based on game state.
        
        Args:
            raw_obs (dict): Raw observation
        
        Returns:
            str: "open", "defend", "3bet", "4bet", "allin"
        """
        raised = raw_obs.get('raised', [0, 0])
        big_blind = raw_obs.get('big_blind', 2)
        pot = raw_obs.get('pot', 0)
        
        # In heads-up:
        # - Player 0 is button (acts first preflop)
        # - Player 1 is big blind (acts second preflop, this agent)
        # - We are always player 1 (big blind)
        
        opponent_raised = raised[0] if len(raised) > 0 else 0
        our_raised = raised[1] if len(raised) > 1 else 0
        
        # CRITICAL FIX: Use pot size as fallback when raised array is not updated correctly
        # This happens when the raised array shows [0, 0] even though button has raised
        # Normal pot after button open = SB (1) + BB (2) + raise (6) = 9 chips = 4.5 BB
        # If pot >= 4BB, button has likely raised
        pot_bb = pot / big_blind if big_blind > 0 else 0
        
        # Determine action type based on betting sequence
        # Button open = 3BB total (opponent_raised = 3BB = 6 chips with BB=2)
        # 3-bet = 10BB total (opponent_raised = 10BB = 20 chips)
        # 4-bet = 15-18BB total (opponent_raised = 15-18BB = 30-36 chips)
        
        # CRITICAL FIX: When raised array is [0, 0] but pot indicates a raise, infer from pot
        # Normal pot after button open = SB (1) + BB (2) + raise (6) = 9 chips = 4.5 BB
        # If pot >= 4BB and raised=[0,0], button has raised but raised array wasn't updated
        # Infer opponent_raised from pot: pot = SB + BB + opponent_raise
        # opponent_raise = pot - SB - BB = pot - 3 chips = pot - 1.5 BB
        if opponent_raised <= big_blind * 1.1 and pot_bb >= 4.0:
            # Raised array is unreliable, infer from pot
            # Pot = SB (0.5BB) + BB (1BB) + opponent_raise
            # opponent_raise_bb = pot_bb - 1.5
            inferred_opponent_raised_bb = pot_bb - 1.5
            opponent_raised_bb = inferred_opponent_raised_bb
            # Update opponent_raised for consistency
            opponent_raised = opponent_raised_bb * big_blind
        else:
            # Use raised array normally
            opponent_raised_bb = opponent_raised / big_blind if big_blind > 0 else 0
        
        # Check if button has raised by looking at both raised amount AND pot size
        button_has_raised = opponent_raised > big_blind * 1.1 or pot_bb >= 4.0
        
        # CRITICAL FIX: Big blind should NEVER "open" - they can only check or defend
        # If pot is just blinds (3 chips = 1.5BB) and raised=[0,0], button has checked
        # In this case, big blind should check (not open)
        # If button has raised, we're defending (not opening)
        if not button_has_raised:
            # Button hasn't raised - check if this is just blinds or button checked
            if pot_bb <= 2.0:
                # Just blinds posted - button will act first, we shouldn't be acting yet
                # But if we are acting, button must have checked, so we should check too
                # Return 'defend' with call action (which becomes check)
                return 'defend'
            else:
                # This shouldn't happen, but if pot > 2BB and no raise, something is wrong
                # Default to defend to be safe
                return 'defend'
        elif our_raised <= big_blind * 1.1:
            # Opponent raised, we haven't - determine if we're facing open, 3-bet, or 4-bet
            # Use both raised amount and pot size to determine (pot size is more reliable)
            
            # FIXED: Use more precise thresholds to avoid misclassifying button opens as 3bets
            # Button open = 2.5-3BB (opponent_raised_bb = 2.5-3.0)
            # 3-bet = 7-10BB (opponent_raised_bb = 7-10)
            # 4-bet = 15-18BB (opponent_raised_bb = 15-18)
            
            if opponent_raised_bb >= 15 or pot_bb >= 20:
                # Facing a 4-bet (15BB+)
                return '4bet'
            elif opponent_raised_bb >= 6.5 or pot_bb >= 12:
                # Facing a 3-bet (6.5BB+ but < 15BB, or pot >= 12BB)
                # Changed from 7BB to 6.5BB to be more precise, and pot from 10BB to 12BB
                return '3bet'
            else:
                # Facing a button open (2.5-3BB) - we're defending
                # This is the most common case - button opens to 3BB
                # Pot will be ~4.5BB (1 + 2 + 6 = 9 chips with BB=2)
                return 'defend'
        else:
            # We've already raised - determine if 3-bet or 4-bet
            if our_raised >= big_blind * 15:
                return '4bet'
            else:
                return '3bet'
    
    def _get_preflop_action(self, hand_str, position, action_type, stack_depth):
        """
        Get preflop action using GTO ranges.
        
        Args:
            hand_str (str): Hand string like "AA", "AKs"
            position (str): "button" or "big_blind"
            action_type (str): "open", "defend", "3bet", "4bet"
            stack_depth (float): Stack depth in big blinds
        
        Returns:
            str: "raise", "call", or "fold"
        """
        if not hand_str:
            return 'fold'
        
        # Use GTO rules to get optimal action
        # Explicitly pass use_frequency=True to ensure mixed strategies work correctly
        gto_action = self.gto_rules.get_preflop_action(
            hand_str, position, action_type, int(stack_depth), use_frequency=True
        )
        
        return gto_action
    
    def _should_all_in_preflop(self, hand_str, position, stack_depth, action_type):
        """
        Determine if we should go all-in preflop.
        
        This should ONLY happen with very short stacks (<= 30 BB) or with premium hands
        at medium stacks (50 BB) that are in the all-in range.
        
        IMPORTANT: Never open-shove all-in at 50 BB or deeper - always use normal opening ranges.
        
        Args:
            hand_str (str): Hand string
            position (str): "button" or "big_blind"
            stack_depth (float): Stack depth in big blinds
            action_type (str): Action type ("open", "defend", "3bet", "4bet")
        
        Returns:
            bool: True if should all-in, False otherwise
        """
        # CRITICAL: Never all-in open or defend at 50 BB or deeper
        # This is the key fix - opponent should never open-shove or defend-shove all-in at 50 BB
        if stack_depth >= 50:
            # Never all-in when opening or defending at 50 BB+
            if action_type in ['open', 'defend']:
                return False
            
            # Only all-in if hand is in the all-in range for this stack depth
            # and we're facing a 3-bet or 4-bet
            if action_type in ['3bet', '4bet']:
                allin_range = self.gto_rules._get_allin_range(stack_depth)
                if hand_str in allin_range and allin_range[hand_str] == 'raise':
                    return True
                else:
                    return False
        
        # For short stacks (<= 30 BB), all-in is more common
        if stack_depth <= 30:
            allin_range = self.gto_rules._get_allin_range(stack_depth)
            if hand_str in allin_range and allin_range[hand_str] == 'raise':
                # Still don't open-shove unless it's a very short stack (< 20 BB)
                if action_type == 'open' and stack_depth < 20:
                    return True
                elif action_type in ['3bet', '4bet']:
                    return True
                else:
                    return False
            else:
                return False

        return False
    
    def _get_postflop_action(self, raw_obs, hand_str):
        """
        Get postflop action using simple strategy.
        
        Args:
            raw_obs (dict): Raw observation
            hand_str (str): Hand string
        
        Returns:
            str: Action to take ('raise', 'call', 'fold')
        """
        # Get game state info
        raised = raw_obs.get('raised', [0, 0])
        pot = raw_obs.get('pot', 0)
        big_blind = raw_obs.get('big_blind', 2)
        public_cards = raw_obs.get('public_cards', [])
        
        # Determine if facing a bet
        opponent_raised = raised[0] if len(raised) > 0 else 0
        our_raised = raised[1] if len(raised) > 1 else 0
        facing_bet = opponent_raised > our_raised

        # Simple postflop strategy:
        # - Strong hands: bet/raise
        # - Medium hands: check/call
        # - Weak hands: fold
        
        # For now, use a simple heuristic based on hand strength
        # Premium hands: AA, KK, QQ, JJ, AK
        premium_hands = ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo']
        
        if hand_str in premium_hands:
            # Strong hand - bet/raise if first to act, call if facing bet
            return 'raise'
        
        # Medium hands: pairs, suited connectors, etc.
        # For simplicity, check/call with medium hands
        # TODO: Add proper hand strength evaluation and fold logic
        return 'call'
    
    def step(self, state):
        """
        Predict action given current state.
        
        Args:
            state (dict): State dictionary with 'raw_obs' and 'raw_legal_actions'
        
        Returns:
            action: Action enum value (Action.FOLD, Action.CHECK_CALL, etc.)
        """
        raw_obs = state.get('raw_obs', {})
        raw_legal_actions = state.get('raw_legal_actions', [])

        # Initialize logger
        import logging
        logger = logging.getLogger(__name__)

        if not raw_legal_actions:
            # No legal actions - shouldn't happen
            return Action.FOLD
        
        # Try to determine player_id from state if not set
        # In RLCard, the state might have player_id info
        # For now, we'll infer it from context
        
        # Get hand
        hand = raw_obs.get('hand', [])
        if not hand or len(hand) != 2:
            # No hand or invalid hand - fold
            if Action.FOLD in raw_legal_actions:
                return Action.FOLD
            return raw_legal_actions[0]  # Default to first action
        
        # Convert hand to string
        hand_str = self._hand_to_string(hand)
        if not hand_str:
            # Invalid hand - fold
            if Action.FOLD in raw_legal_actions:
                return Action.FOLD
            return raw_legal_actions[0]
        
        # Get game state info
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        
        position = self._get_position(raw_obs)
        stack_depth = self._get_stack_depth(raw_obs)
        
        # Preflop decision
        if stage == 0:
            action_type = self._get_preflop_action_type(raw_obs)
            
            # CRITICAL FIX: Only bypass GTO lookup for all-in in very specific scenarios
            # NEVER bypass for 'defend' scenarios - always use GTO lookup to get call/fold decisions
            should_all_in = False
            if action_type == 'defend':
                # NEVER all-in when defending - always use GTO lookup to get proper call/fold decisions
                should_all_in = False
            elif action_type == 'open' and stack_depth >= 20:
                # Never all-in open at 20BB+ - always use normal opening ranges
                should_all_in = False
            else:
                # Only check all-in for 3bet/4bet scenarios or very short stack opens
                should_all_in = self._should_all_in_preflop(hand_str, position, stack_depth, action_type)
            
            # ONLY go all-in if explicitly should, and ONLY for very short stacks or premium hands facing 3bet/4bet
            if should_all_in:
                if Action.ALL_IN in raw_legal_actions:
                    return Action.ALL_IN
                elif Action.RAISE_POT in raw_legal_actions:
                    # If all-in not available, use pot raise
                    return Action.RAISE_POT
                else:
                    # If neither all-in nor pot raise available, fall through to GTO lookup
                    logger.warning(f"[GTO AGENT] All-in check returned True but ALL_IN/RAISE_POT not available, falling through to GTO lookup")
            
            # Get GTO action
            gto_action = self._get_preflop_action(hand_str, position, action_type, stack_depth)
            
            # DEBUG: Log GTO action with full context
            
            # CRITICAL DEBUG: Check if GTO action is always 'raise'
            if gto_action == 'raise':
                logger.warning(f"[GTO AGENT] ⚠️ GTO lookup returned 'raise' - checking if this is correct for hand {hand_str} in {action_type} scenario")
            elif gto_action == 'call':
                logger.debug(f"[GTO AGENT] GTO lookup returned 'call' for hand {hand_str} in {action_type} scenario")
            elif gto_action == 'fold':
                logger.debug(f"[GTO AGENT] GTO lookup returned 'fold' for hand {hand_str} in {action_type} scenario")
            else:
                logger.error(f"[GTO AGENT] ✗ GTO lookup returned unexpected value: '{gto_action}'")
            
            # SAFEGUARD: Validate GTO action
            if gto_action not in ['raise', 'call', 'fold']:
                logger.error(f"[GTO AGENT] Invalid GTO action '{gto_action}', defaulting to 'fold'")
                gto_action = 'fold'
            
            # Map GTO action to RLCard Action enum
            if gto_action == 'raise':
                # CRITICAL: Never all-in when opening or defending at 50 BB+
                # Only use normal raise sizes
                if Action.RAISE_HALF_POT in raw_legal_actions:
                    return Action.RAISE_HALF_POT
                elif Action.RAISE_POT in raw_legal_actions:
                    # Only use pot raise if stack depth is reasonable
                    if stack_depth >= 50 and action_type == 'open':
                        # At 50 BB+, don't use pot raise for opening (could be too large)
                        if Action.CHECK_CALL in raw_legal_actions:
                            return Action.CHECK_CALL
                        elif Action.FOLD in raw_legal_actions:
                            return Action.FOLD
                    return Action.RAISE_POT
                elif Action.CHECK_CALL in raw_legal_actions:
                    return Action.CHECK_CALL
                elif Action.FOLD in raw_legal_actions:
                    return Action.FOLD
                else:
                    # Last resort - but NEVER all-in for opening/defending at 50 BB+
                    if stack_depth >= 50 and action_type in ['open', 'defend']:
                        # Prefer check/call or fold over all-in
                        if Action.CHECK_CALL in raw_legal_actions:
                            return Action.CHECK_CALL
                        elif Action.FOLD in raw_legal_actions:
                            return Action.FOLD
                    # Only use all-in as absolute last resort
                    logger.warning(f"[GTO AGENT] No standard actions available, using first legal action: {raw_legal_actions[0]}")
                    return raw_legal_actions[0]
            elif gto_action == 'call':
                if Action.CHECK_CALL in raw_legal_actions:
                    return Action.CHECK_CALL
                else:
                    logger.warning(f"[GTO AGENT] CHECK_CALL not available! Legal actions: {raw_legal_actions}")
                    logger.warning(f"[GTO AGENT] Using first legal action as fallback: {raw_legal_actions[0]}")
                    return raw_legal_actions[0]
            else:  # fold
                if Action.FOLD in raw_legal_actions:
                    return Action.FOLD
                else:
                    logger.warning(f"[GTO AGENT] FOLD not available! Legal actions: {raw_legal_actions}")
                    logger.warning(f"[GTO AGENT] Using first legal action as fallback: {raw_legal_actions[0]}")
                    return raw_legal_actions[0]
        
        # Postflop decision
        else:
            action = self._get_postflop_action(raw_obs, hand_str)
            
            # Convert string action to Action enum
            if action == 'raise':
                if Action.RAISE_HALF_POT in raw_legal_actions:
                    return Action.RAISE_HALF_POT
                elif Action.RAISE_POT in raw_legal_actions:
                    return Action.RAISE_POT
                elif Action.ALL_IN in raw_legal_actions:
                    return Action.ALL_IN
                elif Action.CHECK_CALL in raw_legal_actions:
                    logger.warning(f"[GTO AGENT] Wanted 'raise' but only CHECK_CALL available, returning CHECK_CALL")
                    return Action.CHECK_CALL
                else:
                    logger.warning(f"[GTO AGENT] Wanted 'raise' but no raise actions available, using first legal: {raw_legal_actions[0]}")
                    return raw_legal_actions[0]
            elif action == 'call' or action == 'check':
                if Action.CHECK_CALL in raw_legal_actions:
                    return Action.CHECK_CALL
                else:
                    logger.warning(f"[GTO AGENT] CHECK_CALL not available! Legal actions: {raw_legal_actions}")
                    logger.warning(f"[GTO AGENT] Using first legal action: {raw_legal_actions[0]}")
                    return raw_legal_actions[0]
            else:  # fold
                if Action.FOLD in raw_legal_actions:
                    return Action.FOLD
                else:
                    logger.warning(f"[GTO AGENT] FOLD not available! Legal actions: {raw_legal_actions}")
                    logger.warning(f"[GTO AGENT] Using first legal action: {raw_legal_actions[0]}")
                    return raw_legal_actions[0]
    
    def eval_step(self, state):
        """
        Evaluation step - same as step for GTO agent.
        
        Args:
            state (dict): State dictionary
        
        Returns:
            tuple: (action, info) where info contains action probabilities
        """
        action = self.step(state)
        
        # Create probability distribution (all probability on chosen action)
        raw_legal_actions = state.get('raw_legal_actions', [])
        probs = [0.0] * len(raw_legal_actions)
        
        if action in raw_legal_actions:
            action_idx = raw_legal_actions.index(action)
            probs[action_idx] = 1.0
        
        info = {
            'probs': {raw_legal_actions[i]: probs[i] for i in range(len(raw_legal_actions))}
        }
        
        return action, info

