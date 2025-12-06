"""
Strategy Evaluator

Core evaluation engine that grades player actions against GTO principles.
"""

import threading
import uuid
from typing import Dict, Any, Optional, List
from .gto_rules import GTORules
from .equity_calculator import EquityCalculator
from .chatbot_coach import ChatbotCoach
from .pattern_recognizer import PatternRecognizer


class StrategyEvaluator:
    """
    Evaluates player actions against GTO strategy principles.
    
    Provides A-F grades with explanations and optimal action recommendations.
    """
    
    def __init__(self):
        """Initialize strategy evaluator with GTO rules and equity calculator."""
        self.gto_rules = GTORules()
        self.equity_calculator = EquityCalculator()
        # Initialize ChatbotCoach and PatternRecognizer for Phase 4 enhancements
        self.chatbot_coach = ChatbotCoach()
        self.pattern_recognizer = PatternRecognizer()
        # Storage for async LLM results: {analysis_id: enhanced_content}
        self.async_results: Dict[str, Dict[str, Any]] = {}
    
    def _safe_str(self, value, default='Unknown'):
        """
        Safely convert value to string, escaping format specifiers.
        
        Args:
            value: Value to convert
            default: Default value if conversion fails
        
        Returns:
            str: Safe string representation
        """
        try:
            result = str(value)
            # Escape format specifiers to prevent formatting errors
            result = result.replace('%', '%%').replace('{', '{{').replace('}', '}}')
            return result
        except Exception:
            return default
    
    def evaluate_action(self, game_state, action, hand_history=None):
        """
        Evaluate a player action.
        
        Args:
            game_state (dict): Current game state from RLCard
            action (dict or str): Player's action
            hand_history (list, optional): History of actions in current hand
        
        Returns:
            dict: {
                'grade': str,  # A-F
                'grade_percentage': int,  # 0-100
                'explanation': str,
                'optimal_action': str,
                'context': dict
            }
        """
        try:
            stage = game_state.get('stage', 0)
            
            if stage == 0:
                return self.evaluate_preflop(game_state, action, hand_history)
            else:
                return self.evaluate_postflop(game_state, action, hand_history)
        except Exception as e:
            # Error handling - return default grade
            # Safely format error message to avoid string formatting issues
            try:
                error_msg = str(e)
                # Replace % with %% to prevent formatting issues if error message contains %
                # Also escape any other potential format specifiers
                error_msg = error_msg.replace('%', '%%')
                # Ensure error message is safe for f-string formatting
                error_msg = error_msg.replace('{', '{{').replace('}', '}}')
            except Exception:
                error_msg = 'An error occurred during evaluation'
            
            # Use string concatenation instead of f-string to be extra safe
            try:
                explanation = 'Evaluation error: ' + error_msg
            except Exception:
                explanation = 'Evaluation error: An error occurred during evaluation'
            
            return {
                'grade': 'C',
                'grade_percentage': 50,
                'explanation': explanation,
                'optimal_action': 'Unknown',
                'context': {}
            }
    
    def evaluate_preflop(self, game_state, action, hand_history=None):
        """
        Evaluate preflop action.
        
        Args:
            game_state (dict): Current game state
            action (dict or str): Player's action
            hand_history (list, optional): Action history
        
        Returns:
            dict: Evaluation result
        """
        # Extract player hand
        hand = game_state.get('hand', [])
        if not hand or len(hand) < 2:
            return self._default_evaluation()
        
        # Convert hand to string format
        hand_str = self._convert_hand_to_string(hand)
        if not hand_str:
            return self._default_evaluation()
        
        # Determine position
        position = self._determine_position(game_state)
        
        # Determine action type
        action_type = self._determine_preflop_action_type(game_state, hand_history)
        
        # Get optimal action
        optimal_action = self.determine_optimal_action(game_state)
        
        # Get optimal action from GTO rules
        stack_depth = self._get_stack_depth(game_state)
        optimal_gto = self.gto_rules.get_preflop_action(
            hand_str, position, action_type,
            stack_depth=stack_depth
        )
        
        # Compare player action to optimal
        player_action_str = self._get_action_string(action)
        optimal_action_str = self._convert_gto_action_to_string(optimal_gto, game_state)
        
        # Evaluate bet sizing if action is a raise
        bet_sizing_eval = None
        if 'raise' in player_action_str.lower() or 'bet' in player_action_str.lower():
            bet_sizing_eval = self.evaluate_preflop_bet_sizing(
                action_type, action, game_state, stack_depth
            )
        
        # Calculate grade
        optimality_score = self._calculate_optimality_score(
            player_action_str, optimal_action_str, optimal_gto, bet_sizing_eval
        )
        grade_result = self.calculate_grade(optimality_score, position=position, stack_depth=stack_depth)
        
        # Generate explanation
        explanation = self.generate_explanation({
            'stage': 'preflop',
            'hand': hand_str,
            'position': position,
            'action_type': action_type,
            'player_action': player_action_str,
            'optimal_action': optimal_action_str,
            'grade': grade_result['grade'],
            'optimality_score': optimality_score,
            'bet_sizing': bet_sizing_eval
        })
        
        return {
            'grade': grade_result['grade'],
            'grade_percentage': grade_result['percentage'],
            'explanation': explanation,
            'optimal_action': optimal_action_str,
            'context': {
                'hand': hand_str,
                'position': position,
                'action_type': action_type,
                'stack_depth': stack_depth,
                'bet_sizing': bet_sizing_eval
            }
        }
    
    def evaluate_preflop_bet_sizing(self, action_type, action, game_state, stack_depth):
        """
        Evaluate preflop bet sizing.
        
        Args:
            action_type (str): "open", "3bet", "4bet"
            action (dict or str): Player's action
            game_state (dict): Current game state
            stack_depth (float): Stack depth in big blinds
        
        Returns:
            dict: {
                'grade': str,  # A-F
                'grade_percentage': int,  # 0-100
                'explanation': str,
                'optimal_size': float,
                'actual_size': float
            }
        """
        # Get bet sizing guidelines
        guidelines = self.gto_rules.get_bet_sizing_guidelines(action_type, stack_depth)
        
        # Extract bet size from action
        big_blind = game_state.get('big_blind', 2)
        if isinstance(action, dict):
            bet_amount = action.get('bet_amount', 0)
        else:
            # Try to extract from game state
            raised = game_state.get('raised', [0, 0])
            bet_amount = raised[0] if len(raised) > 0 else 0
        
        bet_size_bb = bet_amount / big_blind if big_blind > 0 else 0
        
        if bet_size_bb == 0:
            return None  # Not a bet
        
        # Compare to guidelines
        min_size = guidelines.get('min_size', 0)
        max_size = guidelines.get('max_size', 0)
        optimal_size = guidelines.get('optimal_size', 0)
        
        # Calculate grade
        if min_size <= bet_size_bb <= max_size:
            # Within acceptable range
            distance_from_optimal = abs(bet_size_bb - optimal_size)
            range_size = max_size - min_size
            
            if distance_from_optimal < range_size * 0.1:
                # Very close to optimal
                grade_percentage = 95
            elif distance_from_optimal < range_size * 0.2:
                # Close to optimal
                grade_percentage = 85
            else:
                # Within range but not optimal
                grade_percentage = 75
        else:
            # Outside acceptable range
            if bet_size_bb < min_size:
                grade_percentage = 50  # Too small
            else:
                grade_percentage = 40  # Too large
        
        grade = self.calculate_grade(grade_percentage)['grade']
        
        # Safely format bet sizing explanation with defensive checks
        try:
            bet_size_bb = float(bet_size_bb) if bet_size_bb is not None else 0.0
            optimal_size = float(optimal_size) if optimal_size is not None else 0.0
            min_size = float(min_size) if min_size is not None else 0.0
            max_size = float(max_size) if max_size is not None else 0.0
            explanation = f"Bet sizing: {bet_size_bb:.1f} BB. Optimal: {optimal_size:.1f} BB (range: {min_size:.1f}-{max_size:.1f} BB)."
        except (ValueError, TypeError) as e:
            # Fallback if formatting fails
            explanation = f"Bet sizing: {bet_size_bb} BB. Optimal: {optimal_size} BB (range: {min_size}-{max_size} BB)."
        
        return {
            'grade': grade,
            'grade_percentage': grade_percentage,
            'explanation': explanation,
            'optimal_size': optimal_size,
            'actual_size': bet_size_bb
        }
    
    def evaluate_postflop_bet_sizing(self, action_type, action, game_state, board_texture, hand_strength):
        """
        Evaluate postflop bet sizing.
        
        Args:
            action_type (str): "cbet", "value_bet", "bluff", "overbet"
            action (dict or str): Player's action
            game_state (dict): Current game state
            board_texture (dict): Board texture analysis
            hand_strength (dict): Hand strength categorization
        
        Returns:
            dict: {
                'grade': str,  # A-F
                'grade_percentage': int,  # 0-100
                'explanation': str,
                'optimal_size': str,  # 'half_pot', 'pot', 'overbet'
                'actual_size': float  # As fraction of pot
            }
        """
        # Get pot size
        pot = game_state.get('pot', 0)
        big_blind = game_state.get('big_blind', 2)
        
        # Extract bet size from action
        if isinstance(action, dict):
            bet_amount = action.get('bet_amount', 0)
        else:
            raised = game_state.get('raised', [0, 0])
            bet_amount = raised[0] if len(raised) > 0 else 0
        
        if bet_amount == 0 or pot == 0:
            return None  # Not a bet or no pot
        
        bet_size_fraction = bet_amount / pot
        
        # Determine optimal sizing based on action type and context
        if action_type == 'cbet':
            # Continuation bet: half pot to pot
            optimal_min = 0.5
            optimal_max = 1.0
            optimal_size = 0.6
        elif action_type == 'value_bet':
            # Value bet: half pot to pot (stronger hands can bet larger)
            category = hand_strength.get('category', 'weak')
            if category == 'top_pair_plus':
                optimal_min = 0.5
                optimal_max = 1.0
                optimal_size = 0.75
            else:
                optimal_min = 0.5
                optimal_max = 0.75
                optimal_size = 0.6
        elif action_type == 'bluff':
            # Bluff: balanced with value sizing
            optimal_min = 0.5
            optimal_max = 1.0
            optimal_size = 0.6
        elif action_type == 'overbet':
            # Overbet: specific scenarios (polarized ranges)
            optimal_min = 1.0
            optimal_max = 2.0
            optimal_size = 1.5
        else:
            # Default
            optimal_min = 0.5
            optimal_max = 1.0
            optimal_size = 0.6
        
        # Adjust based on board texture
        if board_texture and board_texture.get('type') == 'wet':
            # Wet boards: larger sizing acceptable
            optimal_max = min(1.5, optimal_max * 1.2)
        elif board_texture and board_texture.get('type') == 'dry':
            # Dry boards: smaller sizing often better
            optimal_max = max(0.5, optimal_max * 0.9)
        
        # Calculate grade
        if optimal_min <= bet_size_fraction <= optimal_max:
            distance_from_optimal = abs(bet_size_fraction - optimal_size)
            range_size = optimal_max - optimal_min
            
            if distance_from_optimal < range_size * 0.1:
                grade_percentage = 95
            elif distance_from_optimal < range_size * 0.2:
                grade_percentage = 85
            else:
                grade_percentage = 75
        else:
            if bet_size_fraction < optimal_min:
                grade_percentage = 50  # Too small
            else:
                grade_percentage = 40  # Too large
        
        grade = self.calculate_grade(grade_percentage)['grade']
        
        # Convert optimal size to string
        if optimal_size < 0.75:
            optimal_size_str = 'half_pot'
        elif optimal_size < 1.25:
            optimal_size_str = 'pot'
        else:
            optimal_size_str = 'overbet'
        
        # Safely format bet sizing explanation with defensive checks
        try:
            bet_size_fraction = float(bet_size_fraction) if bet_size_fraction is not None else 0.0
            optimal_min = float(optimal_min) if optimal_min is not None else 0.0
            optimal_max = float(optimal_max) if optimal_max is not None else 0.0
            explanation = f"Bet sizing: {bet_size_fraction:.1f}x pot. Optimal: {optimal_size_str} ({optimal_min:.1f}-{optimal_max:.1f}x pot)."
        except (ValueError, TypeError) as e:
            # Fallback if formatting fails
            explanation = f"Bet sizing: {bet_size_fraction}x pot. Optimal: {optimal_size_str} ({optimal_min}-{optimal_max}x pot)."
        
        return {
            'grade': grade,
            'grade_percentage': grade_percentage,
            'explanation': explanation,
            'optimal_size': optimal_size_str,
            'actual_size': bet_size_fraction
        }
    
    def evaluate_postflop(self, game_state, action, hand_history=None):
        """
        Evaluate postflop action.
        
        Args:
            game_state (dict): Current game state
            action (dict or str): Player's action
            hand_history (list, optional): Action history
        
        Returns:
            dict: Evaluation result
        """
        # Extract hand and board
        hand = game_state.get('hand', [])
        board = game_state.get('public_cards', [])
        stage = game_state.get('stage', 0)
        
        if not hand or len(hand) < 2:
            return self._default_evaluation()
        
        # Categorize hand strength
        hand_strength = self.equity_calculator.categorize_hand_strength(hand, board, stage)
        
        # Analyze board texture
        board_texture = self.equity_calculator._analyze_board_texture(board)
        
        # Construct opponent range
        position = self._determine_position(game_state)
        stack_depth = self._get_stack_depth(game_state)
        opponent_range = self.equity_calculator._construct_opponent_range(
            hand_history or [], position, stack_depth, current_stage=stage, board=board
        )
        
        # Calculate full equity if possible
        try:
            equity = self.equity_calculator.calculate_full_equity(
                hand, board, opponent_range, stage
            )
            if equity is None:
                equity = self.equity_calculator.estimate_equity(hand_strength['category'])
        except Exception:
            equity = self.equity_calculator.estimate_equity(hand_strength['category'])
        
        # Calculate pot odds
        pot = game_state.get('pot', 0)
        bet = self._get_bet_amount(game_state, action)
        pot_odds = self.gto_rules.calculate_pot_odds(pot, bet) if bet > 0 else None
        
        # Evaluate continuation betting
        was_preflop_aggressor = self._was_preflop_aggressor(hand_history)
        continuation_bet_eval = None
        if was_preflop_aggressor and stage == 1:  # Flop
            continuation_bet_eval = self.evaluate_continuation_bet(
                hand_strength, board_texture, game_state
            )
        
        # Distinguish value vs bluff
        value_bluff_eval = self.distinguish_value_vs_bluff(
            hand_strength, board_texture, opponent_range, game_state
        )
        
        # Determine optimal action
        optimal_action = self._determine_postflop_optimal_action(
            hand_strength, pot_odds, equity, game_state, board_texture, continuation_bet_eval, value_bluff_eval
        )
        
        # Get player action
        player_action_str = self._get_action_string(action)
        
        # Evaluate bet sizing if action is a bet
        bet_sizing_eval = None
        if 'raise' in player_action_str.lower() or 'bet' in player_action_str.lower():
            # Determine action type for bet sizing
            if continuation_bet_eval and continuation_bet_eval.get('should_cbet'):
                sizing_action_type = 'cbet'
            elif value_bluff_eval and value_bluff_eval.get('type') == 'value':
                sizing_action_type = 'value_bet'
            elif value_bluff_eval and value_bluff_eval.get('type') == 'bluff':
                sizing_action_type = 'bluff'
            else:
                sizing_action_type = 'value_bet'  # Default
            
            bet_sizing_eval = self.evaluate_postflop_bet_sizing(
                sizing_action_type, action, game_state, board_texture, hand_strength
            )
        
        # Calculate grade
        optimality_score = self._calculate_postflop_optimality_score(
            player_action_str, optimal_action, hand_strength, pot_odds, equity,
            board_texture, continuation_bet_eval, value_bluff_eval, bet_sizing_eval
        )
        grade_result = self.calculate_grade(optimality_score, position=position, stack_depth=stack_depth)
        
        # Generate explanation
        try:
            explanation = self.generate_explanation({
                'stage': self._stage_to_string(stage),
                'hand_strength': hand_strength,
                'pot_odds': pot_odds,
                'equity': equity,
                'player_action': player_action_str,
                'optimal_action': optimal_action,
                'grade': grade_result['grade'],
                'optimality_score': optimality_score,
                'board_texture': board_texture,
                'continuation_bet': continuation_bet_eval,
                'value_bluff': value_bluff_eval
            })
        except Exception as e:
            explanation = f"Evaluation completed. Your action was {player_action_str}. Optimal action: {optimal_action}."
        
        return {
            'grade': grade_result['grade'],
            'grade_percentage': grade_result['percentage'],
            'explanation': explanation,
            'optimal_action': optimal_action,
            'context': {
                'hand_strength': hand_strength['category'],
                'hand_description': hand_strength['description'],
                'pot_odds': pot_odds,
                'equity': equity,
                'position': position,
                'stack_depth': stack_depth,
                'board_texture': board_texture,
                'continuation_bet': continuation_bet_eval,
                'value_bluff': value_bluff_eval
            }
        }
    
    def evaluate_continuation_bet(self, hand_strength, board_texture, game_state):
        """
        Evaluate continuation betting decision.
        
        Args:
            hand_strength (dict): Hand strength categorization
            board_texture (dict): Board texture analysis
            game_state (dict): Current game state
        
        Returns:
            dict: {
                'should_cbet': bool,
                'frequency': float,  # 0-100%
                'optimal_size': str,  # 'half_pot', 'pot', 'overbet'
                'explanation': str
            }
        """
        # GTO continuation bet frequency: ~60-70% on flop
        base_frequency = 65.0
        
        # Adjust based on board texture
        if board_texture.get('type') == 'wet':
            # Wet boards: slightly lower frequency (~55-60%)
            frequency = 57.5
        elif board_texture.get('type') == 'dry':
            # Dry boards: higher frequency (~70-75%)
            frequency = 72.5
        else:
            frequency = base_frequency
        
        # Adjust based on hand strength
        category = hand_strength.get('category', 'weak')
        if category == 'top_pair_plus':
            # Strong hands: bet for value
            should_cbet = True
            optimal_size = 'half_pot'
        elif category == 'second_pair':
            # Medium hands: check/call or bet depending on board
            should_cbet = frequency > 50
            optimal_size = 'half_pot' if should_cbet else 'check'
        elif category == 'draw':
            # Draws: bet for equity
            should_cbet = True
            optimal_size = 'half_pot'
        else:  # weak
            # Weak hands: check/fold or bluff on dry boards
            if board_texture.get('type') == 'dry':
                should_cbet = frequency > 50  # Bluff on dry boards
                optimal_size = 'half_pot' if should_cbet else 'check'
            else:
                should_cbet = False
                optimal_size = 'check'
        
        return {
            'should_cbet': should_cbet,
            'frequency': frequency,
            'optimal_size': optimal_size,
            'explanation': f"Continuation bet frequency: {frequency:.0f}%. Optimal: {optimal_size}."
        }
    
    def distinguish_value_vs_bluff(self, hand_strength, board_texture, opponent_range, game_state):
        """
        Distinguish and evaluate value betting vs bluffing.
        
        Args:
            hand_strength (dict): Hand strength categorization
            board_texture (dict): Board texture analysis
            opponent_range (dict): Opponent's range
            game_state (dict): Current game state
        
        Returns:
            dict: {
                'type': str,  # 'value', 'bluff', 'check'
                'hand_strength_relative': str,  # 'strong', 'medium', 'weak'
                'range_balance': str,  # 'balanced', 'value_heavy', 'bluff_heavy'
                'explanation': str
            }
        """
        category = hand_strength.get('category', 'weak')
        
        # Determine if hand is value or bluff
        if category == 'top_pair_plus':
            bet_type = 'value'
            hand_strength_relative = 'strong'
        elif category == 'second_pair':
            # Medium hands: usually check, but can value bet on dry boards
            if board_texture.get('type') == 'dry':
                bet_type = 'value'
                hand_strength_relative = 'medium'
            else:
                bet_type = 'check'
                hand_strength_relative = 'medium'
        elif category == 'draw':
            # Draws: can be value (strong draws) or bluff (weak draws)
            draws = board_texture.get('draws', [])
            if len(draws) >= 2:  # Combo draw
                bet_type = 'value'
                hand_strength_relative = 'medium'
            else:
                bet_type = 'bluff'
                hand_strength_relative = 'weak'
        else:  # weak
            # Weak hands: usually check/fold, but can bluff on dry boards
            if board_texture.get('type') == 'dry':
                bet_type = 'bluff'
                hand_strength_relative = 'weak'
            else:
                bet_type = 'check'
                hand_strength_relative = 'weak'
        
        # Range balance analysis (simplified)
        # In GTO, value:bluff ratio is typically 2:1 or 3:1
        range_balance = 'balanced'
        
        explanation = f"Hand strength: {hand_strength_relative}. Bet type: {bet_type}."
        if bet_type == 'value':
            explanation += " This hand is strong enough to bet for value."
        elif bet_type == 'bluff':
            explanation += " This hand can be used as a bluff on this board texture."
        else:
            explanation += " This hand is best checked."
        
        return {
            'type': bet_type,
            'hand_strength_relative': hand_strength_relative,
            'range_balance': range_balance,
            'explanation': explanation
        }
    
    def _was_preflop_aggressor(self, hand_history):
        """Check if player was the preflop aggressor."""
        if not hand_history:
            return False
        
        for entry in hand_history:
            if isinstance(entry, dict):
                player_id = entry.get('player_id', -1)
                action = entry.get('action', '').lower()
                if player_id == 0 and ('raise' in action or '3-bet' in action or '4-bet' in action):
                    return True
        
        return False
    
    def calculate_grade(self, optimality_score, position=None, stack_depth=None):
        """
        Calculate A-F grade from optimality score with context adjustments.
        
        Args:
            optimality_score (float): Score from 0-100
            position (str, optional): "button" or "big_blind" for position adjustments
            stack_depth (float, optional): Stack depth for adjustments
        
        Returns:
            dict: {'grade': str, 'percentage': int}
        """
        # Clamp to 0-100 and round to 1 decimal for precision
        score = round(max(0.0, min(100.0, float(optimality_score))), 1)
        
        # Apply position adjustments (subtle)
        if position == 'button':
            # Button: slightly more lenient (+2-3%)
            score = min(100, score + 2.5)
        elif position == 'big_blind':
            # Big blind: slightly stricter (-2-3%)
            score = max(0, score - 2.5)
        
        # Apply stack depth adjustments
        if stack_depth is not None:
            if stack_depth < 30:
                # Shorter stacks: more lenient for marginal decisions (+3-5%)
                score = min(100, score + 4.0)
            elif stack_depth > 100:
                # Deeper stacks: stricter for marginal decisions (-3-5%)
                score = max(0, score - 4.0)
        
        # Round to integer for final percentage
        score_int = int(round(score))
        
        # Clear grade boundaries with consistent rounding
        if score_int >= 90:
            return {'grade': 'A', 'percentage': score_int}
        elif score_int >= 80:
            return {'grade': 'B', 'percentage': score_int}
        elif score_int >= 70:
            return {'grade': 'C', 'percentage': score_int}
        elif score_int >= 60:
            return {'grade': 'D', 'percentage': score_int}
        else:
            return {'grade': 'F', 'percentage': score_int}
    
    def generate_explanation(self, evaluation_data):
        """
        Generate rule-based explanation for evaluation.
        
        Args:
            evaluation_data (dict): Evaluation data
        
        Returns:
            str: Explanation text
        """
        stage = evaluation_data.get('stage', 'unknown')
        
        if stage == 'preflop':
            return self._generate_preflop_explanation(evaluation_data)
        else:
            return self._generate_postflop_explanation(evaluation_data)
    
    def determine_optimal_action(self, game_state):
        """
        Determine optimal action for current game state.
        
        Args:
            game_state (dict): Current game state
        
        Returns:
            dict: {'action': str, 'explanation': str}
        """
        stage = game_state.get('stage', 0)
        
        if stage == 0:
            return self._determine_preflop_optimal_action(game_state)
        else:
            hand = game_state.get('hand', [])
            board = game_state.get('public_cards', [])
            hand_strength = self.equity_calculator.categorize_hand_strength(hand, board, stage)
            pot = game_state.get('pot', 0)
            bet = self._get_current_bet(game_state)
            pot_odds = self.gto_rules.calculate_pot_odds(pot, bet) if bet > 0 else None
            equity = self.equity_calculator.estimate_equity(hand_strength['category'])
            
            optimal = self._determine_postflop_optimal_action(
                hand_strength, pot_odds, equity, game_state
            )
            return {'action': optimal, 'explanation': f"Optimal play based on {hand_strength['description']}"}
    
    # Helper methods
    
    def _convert_hand_to_string(self, hand):
        """Convert RLCard hand to string format (e.g., 'AA', 'AKs')."""
        if len(hand) < 2:
            return None
        
        # Convert card indices to ranks
        ranks = []
        suits = []
        for card in hand:
            # Convert to int if it's a string or numpy type
            if not isinstance(card, int):
                try:
                    card = int(card)
                except (ValueError, TypeError):
                    card = 0  # Safe fallback
            rank = card % 13
            suit = card // 13
            ranks.append(rank)
            suits.append(suit)
        
        ranks.sort(reverse=True)
        
        # Convert ranks to card names
        rank_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        card1 = rank_names[ranks[0]]
        card2 = rank_names[ranks[1]]
        
        # Check if pair
        if ranks[0] == ranks[1]:
            return f"{card1}{card1}"
        
        # Check if suited
        if suits[0] == suits[1]:
            return f"{card1}{card2}s"
        else:
            return f"{card1}{card2}o"
    
    def _determine_position(self, game_state):
        """Determine player position (button or big_blind)."""
        # In HUNL, button is on the small blind position
        # Use button_id if available, otherwise calculate from dealer_id
        button_id = game_state.get('button_id')
        if button_id is None:
            # Fallback: calculate button position from dealer_id
            dealer_id = game_state.get('dealer_id', 0)
            num_players = 2  # HUNL
            button_id = (dealer_id + 1) % num_players
        
        current_player = game_state.get('current_player', 0)
        
        # In heads-up, button is on small blind, big blind is other player
        if current_player == button_id:
            return 'button'
        else:
            return 'big_blind'
    
    def _determine_preflop_action_type(self, game_state, hand_history):
        """Determine preflop action type (open, defend, 3bet, 4bet)."""
        if not hand_history:
            return 'open'
        
        # Count raises in history
        raise_count = 0
        for entry in hand_history:
            action = entry.get('action', '')
            if 'raise' in action.lower() or '3-bet' in action.lower() or '4-bet' in action.lower():
                raise_count += 1
        
        if raise_count >= 3:
            return '4bet'
        elif raise_count >= 2:
            return '3bet'
        elif raise_count >= 1:
            return 'defend'
        else:
            return 'open'
    
    def _get_stack_depth(self, game_state):
        """Get stack depth in big blinds."""
        big_blind = game_state.get('big_blind', 2)
        all_chips = game_state.get('all_chips', [0, 0])
        player_chips = all_chips[0] if len(all_chips) > 0 else 0
        
        if big_blind == 0:
            return 100  # Default
        
        return player_chips / big_blind
    
    def _get_action_string(self, action):
        """Convert action to string."""
        if isinstance(action, dict):
            return action.get('action', 'Unknown')
        elif isinstance(action, str):
            return action
        else:
            return 'Unknown'
    
    def _convert_gto_action_to_string(self, gto_action, game_state):
        """Convert GTO action (raise/call/fold) to readable string."""
        if gto_action == 'fold':
            return 'Fold'
        elif gto_action == 'call':
            return 'Check/Call'
        elif gto_action == 'raise':
            # Default to ½ pot raise
            return 'Raise ½ Pot'
        else:
            return 'Unknown'
    
    def _calculate_optimality_score(self, player_action, optimal_action, optimal_gto, bet_sizing_eval=None):
        """Calculate optimality score (0-100) for preflop."""
        # Simple scoring: exact match = 100, close = 80-90, different = lower
        player_lower = player_action.lower()
        optimal_lower = optimal_action.lower()
        
        if player_lower == optimal_lower:
            base_score = 100
        elif 'fold' in player_lower and 'fold' in optimal_lower:
            base_score = 100
        elif 'raise' in player_lower and optimal_gto == 'raise':
            base_score = 85  # Close but may be wrong sizing
        elif 'call' in player_lower and optimal_gto == 'call':
            base_score = 90
        elif optimal_gto == 'fold' and 'fold' not in player_lower:
            base_score = 30  # Should have folded
        elif optimal_gto == 'raise' and 'raise' not in player_lower:
            base_score = 50  # Should have raised
        else:
            base_score = 40  # Suboptimal
        
        # Adjust based on bet sizing
        if bet_sizing_eval:
            sizing_grade_pct = bet_sizing_eval.get('grade_percentage', 50)
            # Weight bet sizing at 20% of total score
            base_score = base_score * 0.8 + sizing_grade_pct * 0.2
        
        return max(0, min(100, base_score))
    
    def _get_bet_amount(self, game_state, action):
        """Get bet amount from action or game state."""
        if isinstance(action, dict):
            return action.get('bet_amount', 0)
        
        # Try to get from raised amounts
        raised = game_state.get('raised', [0, 0])
        if len(raised) > 0:
            return raised[0]
        
        return 0
    
    def _get_current_bet(self, game_state):
        """Get current bet amount to call."""
        raised = game_state.get('raised', [0, 0])
        if len(raised) >= 2:
            return max(0, raised[1] - raised[0])
        return 0
    
    def _determine_postflop_optimal_action(self, hand_strength, pot_odds, equity, game_state,
                                          board_texture=None, continuation_bet_eval=None, value_bluff_eval=None):
        """Determine optimal postflop action."""
        category = hand_strength['category']
        
        # Use continuation bet evaluation if available
        if continuation_bet_eval and continuation_bet_eval.get('should_cbet'):
            optimal_size = continuation_bet_eval.get('optimal_size', 'half_pot')
            if optimal_size == 'half_pot':
                return 'Raise ½ Pot'
            elif optimal_size == 'pot':
                return 'Raise Pot'
            elif optimal_size == 'overbet':
                return 'Raise Overbet'
            else:
                return 'Check/Call'
        
        # Use value/bluff evaluation if available
        if value_bluff_eval:
            bet_type = value_bluff_eval.get('type', 'check')
            if bet_type == 'value':
                return 'Raise ½ Pot'
            elif bet_type == 'bluff':
                return 'Raise ½ Pot'  # Bluff sizing matches value sizing
            else:  # check
                if pot_odds and equity / 100 > 1 / (pot_odds + 1):
                    return 'Check/Call'
                else:
                    return 'Fold'
        
        # Fallback to original logic
        if category == 'top_pair_plus':
            # Strong hand, bet for value
            return 'Raise ½ Pot'
        elif category == 'second_pair':
            # Medium hand, check/call or fold depending on pot odds
            if pot_odds and equity / 100 > 1 / (pot_odds + 1):
                return 'Check/Call'
            else:
                return 'Fold'
        elif category == 'draw':
            # Draw, check/call if pot odds are good
            if pot_odds and equity / 100 > 1 / (pot_odds + 1):
                return 'Check/Call'
            else:
                return 'Fold'
        else:  # weak
            # Weak hand, fold
            return 'Fold'
    
    def _calculate_postflop_optimality_score(self, player_action, optimal_action, hand_strength, pot_odds, equity,
                                            board_texture=None, continuation_bet_eval=None, value_bluff_eval=None,
                                            bet_sizing_eval=None):
        """Calculate optimality score for postflop."""
        player_lower = player_action.lower()
        optimal_lower = optimal_action.lower()
        
        base_score = 0
        
        # Exact match
        if player_lower == optimal_lower:
            base_score = 100
        elif 'fold' in player_lower and 'fold' in optimal_lower:
            base_score = 100
        elif 'call' in player_lower and 'call' in optimal_lower:
            base_score = 90
        elif 'raise' in player_lower and 'raise' in optimal_lower:
            base_score = 90
        elif optimal_lower == 'fold' and 'fold' not in player_lower:
            base_score = 30  # Should have folded
        elif 'raise' in optimal_lower and 'raise' not in player_lower:
            base_score = 60  # Should have bet
        else:
            base_score = 50  # Suboptimal
        
        # Adjust based on continuation betting
        if continuation_bet_eval:
            if continuation_bet_eval.get('should_cbet') and 'raise' not in player_lower:
                base_score -= 10  # Penalty for missing continuation bet
        
        # Adjust based on value/bluff evaluation
        if value_bluff_eval:
            bet_type = value_bluff_eval.get('type', 'check')
            if bet_type == 'value' and 'raise' not in player_lower:
                base_score -= 15  # Penalty for missing value bet
            elif bet_type == 'bluff' and 'raise' not in player_lower:
                base_score -= 5  # Smaller penalty for missing bluff
        
        # Adjust based on bet sizing
        if bet_sizing_eval:
            sizing_grade_pct = bet_sizing_eval.get('grade_percentage', 50)
            # Weight bet sizing at 20% of total score
            base_score = base_score * 0.8 + sizing_grade_pct * 0.2
        
        return max(0, min(100, base_score))  # Clamp to 0-100
    
    def _stage_to_string(self, stage):
        """Convert stage number to string."""
        stages = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        return stages.get(stage, 'unknown')
    
    def _generate_preflop_explanation(self, data):
        """Generate enhanced preflop explanation with strategic reasoning."""
        try:
            # Use safe string conversion to prevent formatting errors
            hand = self._safe_str(data.get('hand', 'Unknown'))
            position = self._safe_str(data.get('position', 'unknown'))
            action_type = self._safe_str(data.get('action_type', 'unknown'))
            player_action = self._safe_str(data.get('player_action', 'Unknown'))
            optimal_action = self._safe_str(data.get('optimal_action', 'Unknown'))
            grade = self._safe_str(data.get('grade', 'C'))
            bet_sizing = data.get('bet_sizing')
            stack_depth = data.get('stack_depth', 100)
            
            # Build explanation with context
            explanation_parts = []
            
            # Position context
            position_advantage = "Button position gives you the advantage of acting last postflop" if position == 'button' else "Big blind position means you act first postflop"
            explanation_parts.append(f"From {position} position ({position_advantage.lower()}).")
            
            # Hand and action context
            explanation_parts.append(f"You {player_action} with {hand}.")
            
            # Stack depth context
            if stack_depth < 30:
                explanation_parts.append(f"With {stack_depth:.0f} BB stack depth, ranges are wider due to shorter stack dynamics.")
            elif stack_depth > 100:
                explanation_parts.append(f"With {stack_depth:.0f} BB stack depth, ranges are tighter due to deeper stack dynamics.")
            
            # Strategic reasoning - enhanced with more context
            if grade in ['A', 'B']:
                explanation_parts.append(f"This is optimal play - {hand} is in the optimal {position} {action_type} range.")
                if action_type == 'open':
                    explanation_parts.append("Opening this hand allows you to build the pot in position with a strong range, giving you control of the hand postflop.")
                elif action_type == 'defend':
                    explanation_parts.append("Defending with this hand is profitable given the pot odds and your position, allowing you to realize your equity effectively.")
                elif action_type in ['3bet', '4bet']:
                    explanation_parts.append(f"This {action_type} puts pressure on your opponent and builds the pot with a strong hand, maximizing value when you have the best hand.")
            else:
                explanation_parts.append(f"The optimal play would have been {optimal_action}.")
                if action_type == 'open':
                    explanation_parts.append(f"{hand} is not in the optimal opening range from {position} - folding preserves your stack for better spots where you have a stronger edge.")
                elif action_type == 'defend':
                    explanation_parts.append(f"{hand} is not strong enough to defend against this {action_type} - folding is more profitable as you'll often be dominated or outdrawn.")
                else:
                    explanation_parts.append(f"{hand} is not in the optimal {action_type} range for this situation - the pot odds and position don't justify this action.")
            
            # Bet sizing feedback
            if bet_sizing:
                sizing_grade = bet_sizing.get('grade', 'C')
                if sizing_grade not in ['A', 'B']:
                    bet_sizing_explanation = bet_sizing.get('explanation', '')
                    if bet_sizing_explanation:
                        # Ensure bet_sizing_explanation is a string and doesn't cause formatting issues
                        bet_sizing_explanation = self._safe_str(bet_sizing_explanation, 'Bet sizing could be improved')
                        explanation_parts.append(f"Bet sizing note: {bet_sizing_explanation}")
            
            return " ".join(explanation_parts)
        except Exception as e:
            # Safely handle exceptions to avoid string formatting errors
            player_action = self._safe_str(data.get('player_action', 'Unknown'))
            optimal_action = self._safe_str(data.get('optimal_action', 'Unknown'))
            # Use string concatenation to be extra safe
            try:
                return "Preflop evaluation completed. Action: " + player_action + ", Optimal: " + optimal_action + "."
            except Exception:
                return "Preflop evaluation completed. Action: Unknown, Optimal: Unknown."
    
    def _generate_postflop_explanation(self, data):
        """Generate enhanced postflop explanation with strategic reasoning."""
        hand_strength = data.get('hand_strength', {})
        pot_odds = data.get('pot_odds')
        equity = data.get('equity', 0)
        player_action = data.get('player_action', 'Unknown')
        optimal_action = data.get('optimal_action', 'Unknown')
        grade = data.get('grade', 'C')
        board_texture = data.get('board_texture', {})
        continuation_bet = data.get('continuation_bet')
        value_bluff = data.get('value_bluff')
        bet_sizing = data.get('bet_sizing')
        
        # Ensure equity is a number
        try:
            equity = float(equity) if equity is not None else 0
        except (ValueError, TypeError):
            equity = 0
        
        desc = hand_strength.get('description', 'Unknown hand') if isinstance(hand_strength, dict) else str(hand_strength)
        
        # Format pot odds safely
        try:
            po_text = f"{float(pot_odds):.1f}:1" if pot_odds is not None else "N/A"
        except (ValueError, TypeError):
            po_text = "N/A"
        
        explanation_parts = []
        
        # Hand strength context
        explanation_parts.append(f"You {player_action} with {desc}.")
        
        # Board texture analysis
        if board_texture:
            texture_type = board_texture.get('type', 'unknown')
            draws = board_texture.get('draws', [])
            coordination = board_texture.get('coordination', 'unknown')
            
            if texture_type == 'wet':
                explanation_parts.append("This is a wet board with many draws available, making it more dangerous for marginal hands.")
            elif texture_type == 'dry':
                explanation_parts.append("This is a dry board with few draws, making it safer for value betting with strong hands.")
            elif texture_type == 'dynamic':
                explanation_parts.append("This is a dynamic board that can change significantly on later streets, requiring careful consideration of future action.")
            else:
                explanation_parts.append("This is a static board unlikely to change much, making it easier to evaluate hand strength.")
            
            if draws:
                explanation_parts.append(f"Available draws: {', '.join(draws)} - these can improve your opponent's hand on later streets.")
        
        # Equity and pot odds
        explanation_parts.append(f"Your estimated equity is {equity:.0f}% and pot odds are {po_text}.")
        
        # Continuation betting analysis
        if continuation_bet:
            cbet_freq = continuation_bet.get('frequency', 0)
            # Ensure cbet_freq is a number
            try:
                cbet_freq = float(cbet_freq) if cbet_freq is not None else 0.0
            except (ValueError, TypeError):
                cbet_freq = 0.0
            should_cbet = continuation_bet.get('should_cbet', False)
            if should_cbet:
                explanation_parts.append(f"Continuation betting is optimal here (frequency: {cbet_freq:.0f}%) - this allows you to build the pot with strong hands and apply pressure with bluffs.")
            else:
                explanation_parts.append(f"Checking is preferred here (continuation bet frequency: {cbet_freq:.0f}%) - this board texture doesn't favor continuation betting with this hand.")
        
        # Value vs bluff analysis
        if value_bluff:
            bet_type = value_bluff.get('type', 'check')
            if bet_type == 'value':
                explanation_parts.append("This hand is strong enough to bet for value - you want to build the pot when you have the best hand.")
            elif bet_type == 'bluff':
                explanation_parts.append("This hand can be used as a bluff on this board texture - betting here applies pressure and can force folds from better hands.")
            else:
                explanation_parts.append("This hand is best checked given the board texture - betting here would likely be unprofitable.")
        
        # Strategic reasoning - enhanced
        if grade in ['A', 'B']:
            explanation_parts.append("This is good play that aligns with GTO principles and maximizes your expected value in this situation.")
        else:
            explanation_parts.append(f"The optimal play would have been {optimal_action}.")
            if equity > 0 and pot_odds:
                try:
                    # Ensure pot_odds is a number
                    pot_odds_num = float(pot_odds) if pot_odds is not None else None
                    if pot_odds_num is not None and pot_odds_num > 0:
                        required_equity = 1 / (pot_odds_num + 1) * 100
                        if equity < required_equity:
                            explanation_parts.append(f"Your equity ({equity:.0f}%) is below the required equity ({required_equity:.0f}%) to call profitably - calling here loses money in the long run.")
                        else:
                            explanation_parts.append(f"Your equity ({equity:.0f}%) is sufficient to call profitably - calling here is mathematically correct given the pot odds.")
                except (ValueError, TypeError, ZeroDivisionError):
                    # Skip equity comparison if pot_odds is invalid
                    pass
        
        # Bet sizing feedback
        if bet_sizing:
            sizing_grade = bet_sizing.get('grade', 'C')
            if sizing_grade not in ['A', 'B']:
                    bet_sizing_explanation = bet_sizing.get('explanation', '')
                    if bet_sizing_explanation:
                        # Ensure bet_sizing_explanation is a string and doesn't cause formatting issues
                        bet_sizing_explanation = self._safe_str(bet_sizing_explanation, 'Bet sizing could be improved')
                        explanation_parts.append(f"Bet sizing: {bet_sizing_explanation}")
        
        return " ".join(explanation_parts)
    
    def _determine_preflop_optimal_action(self, game_state):
        """Determine optimal preflop action."""
        hand = game_state.get('hand', [])
        if not hand or len(hand) < 2:
            return {'action': 'Unknown', 'explanation': 'Cannot determine optimal action'}
        
        hand_str = self._convert_hand_to_string(hand)
        position = self._determine_position(game_state)
        action_type = self._determine_preflop_action_type(game_state, None)
        
        optimal_gto = self.gto_rules.get_preflop_action(
            hand_str, position, action_type,
            stack_depth=self._get_stack_depth(game_state)
        )
        
        optimal_str = self._convert_gto_action_to_string(optimal_gto, game_state)
        return {'action': optimal_str, 'explanation': f"Optimal {position} play with {hand_str}"}
    
    def analyze_hand(self, hand_history, game_state, session_id: Optional[str] = None,
                    hand_history_storage: Optional[Dict] = None):
        """
        Analyze a complete hand with all decisions.
        
        Phase 4 Enhanced: Returns initial rule-based analysis immediately,
        then triggers async LLM enhancements in background.
        
        Args:
            hand_history (list): Complete hand history with all decisions
            game_state (dict): Final game state
            session_id (str, optional): Session ID for pattern recognition
            hand_history_storage (dict, optional): Hand history storage for pattern recognition
        
        Returns:
            dict: {
                'overall_grade': str,
                'overall_grade_percentage': int,
                'decisions': list,
                'key_insights': list,  # Phase 1 rule-based
                'learning_points': list,  # Phase 1 rule-based
                'analysis_id': str,  # For async LLM retrieval
                'llm_enhanced': bool  # Whether LLM content is ready
            }
        """
        decisions = []
        total_score = 0
        decision_count = 0
        
        # Evaluate each player decision in the hand
        for i, decision in enumerate(hand_history):
            if decision.get('player_id') == 0:  # Only evaluate player decisions
                try:
                    # Reconstruct game state at decision point
                    decision_state = self._reconstruct_decision_state(decision, game_state, hand_history[:i+1])
                    
                    # Evaluate the decision
                    action = decision.get('action', 'Unknown')
                    evaluation = self.evaluate_action(decision_state, action, hand_history[:i+1])
                    
                    decisions.append({
                        'stage': self._get_decision_stage(decision, hand_history[:i+1]),
                        'action': action,
                        'grade': evaluation['grade'],
                        'grade_percentage': evaluation['grade_percentage'],
                        'explanation': evaluation['explanation'],
                        'optimal_action': evaluation['optimal_action'],
                        'context': evaluation['context']
                    })
                    
                    total_score += evaluation['grade_percentage']
                    decision_count += 1
                except Exception as e:
                    # Skip decisions that can't be evaluated
                    # logging removed
                    # Safely format error message to avoid string formatting issues
                    try:
                        error_msg = str(e)
                        # Escape format specifiers to prevent formatting errors
                        error_msg = error_msg.replace('%', '%%').replace('{', '{{').replace('}', '}}')
                    except Exception:
                        pass
                    continue
        
        # Calculate overall grade (Phase 4 enhanced with weighting)
        if decision_count > 0:
            average_score = total_score / decision_count
            # Phase 4: Enhanced grade calculation with weighting
            overall_grade = self._calculate_enhanced_overall_grade(decisions, average_score, decision_count)
        else:
            overall_grade = {'grade': 'C', 'percentage': 50}
        
        # Generate Phase 1's rule-based key insights
        key_insights = self._generate_key_insights(decisions, decision_count)
        
        # Generate Phase 1's rule-based learning points
        learning_points = self._generate_learning_points(decisions)
        
        # Build initial analysis (Phase 1 rule-based)
        initial_analysis = {
            'overall_grade': overall_grade['grade'],
            'overall_grade_percentage': overall_grade['percentage'],
            'decisions': decisions,
            'key_insights': key_insights,
            'learning_points': learning_points
        }
        
        # Generate analysis ID for async LLM retrieval
        analysis_id = str(uuid.uuid4())
        
        # Trigger async LLM enhancements in background
        if session_id and hand_history_storage:
            thread = threading.Thread(
                target=self._generate_async_llm_enhancements,
                args=(analysis_id, initial_analysis, game_state, session_id, hand_history_storage),
                daemon=True
            )
            thread.start()
        
        # Return initial response immediately
        return {
            **initial_analysis,
            'analysis_id': analysis_id,
            'llm_enhanced': False  # LLM content not ready yet
        }
    
    def _generate_async_llm_enhancements(self, analysis_id: str, initial_analysis: Dict[str, Any],
                                        game_state: Dict[str, Any], session_id: str,
                                        hand_history_storage: Dict):
        """
        Generate LLM enhancements asynchronously in background thread.
        
        Args:
            analysis_id: Unique analysis ID
            initial_analysis: Initial rule-based analysis
            game_state: Game state
            session_id: Session ID
            hand_history_storage: Hand history storage
        """
        # logging removed
        
        try:
            # Check if API keys are available before attempting LLM calls
            if not self.chatbot_coach.api_key_available or not self.chatbot_coach.client:
                # Mark as ready immediately with API unavailable flag
                self.async_results[analysis_id] = {
                    'llm_insights': [],
                    'enhanced_explanations': {},
                    'pattern_insights': [],
                    'llm_learning_points': [],
                    'ready': True,
                    'api_unavailable': True  # Flag to indicate API keys missing
                }
                return
            
            enhanced_content = {
                'llm_insights': [],
                'enhanced_explanations': {},
                'pattern_insights': [],
                'llm_learning_points': [],
                'ready': False,
                'api_unavailable': False
            }
            
            # 1. Generate LLM-powered insights
            try:
                llm_insights = self.chatbot_coach.generate_hand_insights(initial_analysis, game_state)
                enhanced_content['llm_insights'] = llm_insights
            except Exception as e:
                pass
            
            # 2. Enhance explanations for each decision
            try:
                enhanced_explanations = {}
                for i, decision in enumerate(initial_analysis.get('decisions', [])):
                    decision_context = {
                        'stage': decision.get('stage', 'unknown'),
                        'action': decision.get('action', 'Unknown'),
                        'grade': decision.get('grade', 'C'),
                        'optimal_action': decision.get('optimal_action', 'Unknown')
                    }
                    enhanced_explanation = self.chatbot_coach.enhance_explanation(
                        decision.get('explanation', ''),
                        decision_context,
                        game_state
                    )
                    enhanced_explanations[i] = enhanced_explanation
                enhanced_content['enhanced_explanations'] = enhanced_explanations
            except Exception as e:
                pass
            
            # 3. Pattern recognition
            try:
                if hand_history_storage:
                    recent_hands = self.pattern_recognizer.get_recent_hands(
                        session_id, hand_history_storage, count=10
                    )
                    if len(recent_hands) >= 2:
                        pattern_insights = self.chatbot_coach.analyze_patterns(
                            recent_hands, initial_analysis
                        )
                        enhanced_content['pattern_insights'] = pattern_insights
            except Exception as e:
                pass
            
            # 4. Generate LLM-powered learning points
            try:
                pattern_insights = enhanced_content.get('pattern_insights', [])
                llm_learning_points = self.chatbot_coach.generate_learning_points(
                    initial_analysis, pattern_insights
                )
                enhanced_content['llm_learning_points'] = llm_learning_points
            except Exception as e:
                pass
            
            # Mark as ready
            enhanced_content['ready'] = True
            
            # Store results
            self.async_results[analysis_id] = enhanced_content
            
        except Exception as e:
            import traceback
            # Store empty result to indicate failure
            self.async_results[analysis_id] = {
                'llm_insights': [],
                'enhanced_explanations': {},
                'pattern_insights': [],
                'llm_learning_points': [],
                'ready': True,  # Mark ready even on failure
                'api_unavailable': False,
                'error': str(e)  # Include error message for debugging
            }
    
    def get_async_llm_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get async LLM enhancement results.
        
        Args:
            analysis_id: Analysis ID
        
        Returns:
            dict: Enhanced content or None if not ready
        """
        return self.async_results.get(analysis_id)
    
    def _reconstruct_decision_state(self, decision, final_state, history_up_to_decision):
        """Reconstruct game state at decision point."""
        # Simplified reconstruction - use final state with adjustments
        # In a full implementation, we'd track state at each decision
        stage = self._determine_stage_from_history(history_up_to_decision)
        
        state = final_state.copy()
        state['stage'] = stage
        
        return state
    
    def _determine_stage_from_history(self, history):
        """Determine game stage from action history."""
        # Count actions to determine stage
        # Simplified: assume preflop until board cards appear
        for entry in history:
            if 'flop' in entry.get('action', '').lower():
                return 1
            elif 'turn' in entry.get('action', '').lower():
                return 2
            elif 'river' in entry.get('action', '').lower():
                return 3
        
        return 0  # Preflop
    
    def _get_decision_stage(self, decision, history):
        """Get stage string for decision."""
        stage_num = self._determine_stage_from_history(history)
        return self._stage_to_string(stage_num)
    
    def _generate_key_insights(self, decisions, decision_count):
        """Generate key insights from hand analysis."""
        insights = []
        
        if decision_count == 0:
            return ["No player decisions to analyze"]
        
        # Count optimal decisions
        optimal_count = sum(1 for d in decisions if d['grade'] in ['A', 'B'])
        insights.append(f"You made {optimal_count} optimal decisions out of {decision_count} total decisions")
        
        # Find weakest decision
        if decisions:
            weakest = min(decisions, key=lambda d: d['grade_percentage'])
            insights.append(f"Your weakest decision was {weakest['action']} which received a {weakest['grade']}")
        
        # Count folds
        fold_count = sum(1 for d in decisions if 'fold' in d['action'].lower())
        if fold_count > 0:
            insights.append(f"You showed discipline by folding {fold_count} time(s)")
        
        # Preflop vs postflop performance
        preflop_decisions = [d for d in decisions if d['stage'] == 'preflop']
        postflop_decisions = [d for d in decisions if d['stage'] != 'preflop']
        
        if preflop_decisions:
            avg_preflop = sum(d['grade_percentage'] for d in preflop_decisions) / len(preflop_decisions)
            insights.append(f"Preflop performance: {avg_preflop:.0f}% average")
        
        if postflop_decisions:
            avg_postflop = sum(d['grade_percentage'] for d in postflop_decisions) / len(postflop_decisions)
            insights.append(f"Postflop performance: {avg_postflop:.0f}% average")
        
        return insights
    
    def _generate_learning_points(self, decisions):
        """Generate learning points from hand analysis."""
        learning_points = []
        
        if not decisions:
            return ["No decisions to learn from"]
        
        # Find common mistakes
        low_grade_decisions = [d for d in decisions if d['grade'] in ['D', 'F']]
        
        for decision in low_grade_decisions[:3]:  # Top 3 mistakes
            if decision['stage'] == 'preflop':
                learning_points.append(f"Consider {decision['optimal_action']} with similar hands in {decision['context'].get('position', 'this position')}")
            else:
                hand_desc = decision['context'].get('hand_description', 'this hand')
                learning_points.append(f"With {hand_desc}, the optimal play is typically {decision['optimal_action']}")
        
        # General learning points
        if not low_grade_decisions:
            learning_points.append("Great job! All your decisions were optimal or near-optimal")
        else:
            learning_points.append("Review your lower-graded decisions to identify patterns for improvement")
        
        return learning_points
    
    def _calculate_enhanced_overall_grade(self, decisions: List[Dict[str, Any]], 
                                         average_score: float, decision_count: int) -> Dict[str, Any]:
        """
        Calculate enhanced overall hand grade with decision weighting and difficulty adjustment.
        
        Phase 4 Enhancement: Weight decisions by importance, consider difficulty, account for patterns.
        
        Args:
            decisions: List of decision evaluations
            average_score: Average grade percentage
            decision_count: Number of decisions
        
        Returns:
            dict: {'grade': str, 'percentage': int}
        """
        if decision_count == 0:
            return {'grade': 'C', 'percentage': 50}
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for decision in decisions:
            stage = decision.get('stage', 'preflop')
            grade_pct = decision.get('grade_percentage', 50)
            action = decision.get('action', '').lower()
            
            # Base weight
            if stage == 'preflop':
                weight = 1.0  # Base weight for preflop
            else:
                weight = 1.5  # Higher weight for postflop (more complex)
            
            # Critical decision weighting (all-in, large bets)
            if 'all-in' in action or 'all in' in action:
                weight *= 2.0  # Highest weight for all-in
            elif 'pot' in action or 'overbet' in action:
                weight *= 1.5  # Higher weight for large bets
            
            # Difficulty adjustment (simplified - could be enhanced)
            # Complex decisions get positive adjustment
            context = decision.get('context', {})
            if context.get('stack_depth', 100) < 30 or context.get('stack_depth', 100) > 100:
                # Short stack or deep stack = more complex
                grade_pct += 2  # Small positive adjustment
            
            weighted_score += grade_pct * weight
            total_weight += weight
        
        # Calculate weighted average
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = average_score
        
        # Pattern consistency factor (simplified - would use pattern recognition in full implementation)
        # For now, check if decisions are consistently good or bad
        good_decisions = sum(1 for d in decisions if d.get('grade_percentage', 50) >= 80)
        bad_decisions = sum(1 for d in decisions if d.get('grade_percentage', 50) < 50)
        
        if good_decisions >= decision_count * 0.7:
            # Consistent good play - small bonus
            final_score += 2
        elif bad_decisions >= decision_count * 0.5:
            # Repeated mistakes - small penalty
            final_score -= 2
        
        # Clamp to 0-100
        final_score = max(0, min(100, final_score))
        
        # Convert to grade
        return self.calculate_grade(final_score)
    
    def _default_evaluation(self):
        """Return default evaluation when data is missing."""
        return {
            'grade': 'C',
            'grade_percentage': 50,
            'explanation': 'Unable to evaluate action - missing game state data',
            'optimal_action': 'Unknown',
            'context': {}
        }

