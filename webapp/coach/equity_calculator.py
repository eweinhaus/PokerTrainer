"""
Equity Calculator

This module provides hand strength categorization and equity calculations
for postflop evaluation. Uses Monte Carlo simulation with caching for performance.
"""

import random
from functools import lru_cache
import time
from .gto_rules import GTORules


class EquityCalculator:
    """
    Calculates hand strength and provides equity calculations for postflop evaluation.
    
    Uses Monte Carlo simulation with caching for accurate equity calculations.
    """
    
    def __init__(self, cache_size=1000, monte_carlo_iterations=1000, timeout_ms=500, gto_rules=None):
        """
        Initialize equity calculator.
        
        Args:
            cache_size (int): Maximum cache size for equity calculations
            monte_carlo_iterations (int): Number of iterations for Monte Carlo simulation
            timeout_ms (int): Timeout for equity calculations in milliseconds
            gto_rules (GTORules, optional): GTORules instance for accessing GTO ranges
        """
        self.monte_carlo_iterations = min(monte_carlo_iterations, 2000)  # Cap at 2000 for performance
        self.timeout_ms = timeout_ms
        self._equity_cache = {}
        self._cache_max_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        # Initialize GTORules instance for accessing GTO ranges
        self.gto_rules = gto_rules or GTORules()
    
    def _get_cache_key(self, hand, board, opponent_range_hash):
        """Generate cache key for equity calculation."""
        hand_str = str(sorted(hand))
        board_str = str(sorted(board))
        return f"{hand_str}_{board_str}_{opponent_range_hash}"
    
    def categorize_hand_strength(self, hand, board, stage):
        """
        Categorize hand strength into categories.
        
        Args:
            hand (list): Player's hand as list of card indices (RLCard format)
            board (list): Board cards as list of card indices
            stage (int): Game stage (0=preflop, 1=flop, 2=turn, 3=river)
        
        Returns:
            dict: {
                'category': str,  # 'top_pair_plus', 'second_pair', 'draw', 'weak'
                'description': str  # Human-readable description
            }
        """
        if stage == 0:
            return {
                'category': 'preflop',
                'description': 'Preflop hand'
            }
        
        # Convert RLCard card indices to suit/rank
        hand_cards = self._cards_to_suit_rank(hand)
        board_cards = self._cards_to_suit_rank(board)
        all_cards = hand_cards + board_cards
        
        # Evaluate hand strength
        hand_rank = self._evaluate_hand(hand_cards, board_cards)
        
        # Categorize based on hand rank and board texture
        if hand_rank['rank'] >= 7:  # Two pair or better
            return {
                'category': 'top_pair_plus',
                'description': hand_rank['description']
            }
        elif hand_rank['rank'] == 2:  # One pair (rank 2, not 6)
            # Check if it's top pair or second pair
            pair_rank = hand_rank.get('pair_rank', 0)
            board_ranks = [card[1] for card in board_cards]
            max_board_rank = max(board_ranks) if board_ranks else 0
            
            if pair_rank >= max_board_rank:
                return {
                    'category': 'top_pair_plus',
                    'description': f"Top pair {self._rank_to_name(pair_rank)}"
                }
            else:
                return {
                    'category': 'second_pair',
                    'description': f"Second pair {self._rank_to_name(pair_rank)}"
                }
        elif self._has_draw(all_cards):
            draw_type = self._get_draw_type(all_cards)
            return {
                'category': 'draw',
                'description': draw_type
            }
        else:
            return {
                'category': 'weak',
                'description': 'Weak hand (high card or nothing)'
            }
    
    def estimate_equity(self, hand_strength_category, opponent_range=None):
        """
        Estimate equity based on hand strength category.
        
        Args:
            hand_strength_category (str): Category from categorize_hand_strength
            opponent_range (dict, optional): Opponent's range (not used in Phase 1)
        
        Returns:
            float: Estimated equity percentage (0-100)
        """
        # Rule-based estimates
        estimates = {
            'top_pair_plus': 70,  # 60-80% range, use 70% as default
            'second_pair': 40,    # 30-50% range, use 40% as default
            'draw': 25,           # 15-40% range, use 25% as default
            'weak': 15,           # 10-25% range, use 15% as default
            'preflop': 50         # Preflop, use 50% as neutral
        }
        
        return estimates.get(hand_strength_category, 20)
    
    def calculate_full_equity(self, hand, board, opponent_range, stage=1):
        """
        Calculate full equity using Monte Carlo simulation.
        
        Args:
            hand (list): Player's hand as list of card indices (RLCard format)
            board (list): Board cards as list of card indices
            opponent_range (dict): Opponent's range (hand -> probability)
            stage (int): Game stage (1=flop, 2=turn, 3=river)
        
        Returns:
            float: Equity percentage (0-100), or None if calculation fails
        """
        try:
            if not hand or len(hand) < 2:
                return None
            
            # Validate board state
            if board:
                if len(board) < 3 or len(board) > 5:
                    # Invalid board state
                    return None
                if len(set(hand + board)) != len(hand) + len(board):
                    # Duplicate cards
                    return None
            
            if not board:
                # Preflop - use estimate
                return 50.0
            
            # Check cache first
            opponent_range_hash = hash(str(sorted(opponent_range.items()))) if opponent_range else 0
            cache_key = self._get_cache_key(hand, board, opponent_range_hash)
            
            if cache_key in self._equity_cache:
                self._cache_hits += 1
                return self._equity_cache[cache_key]
            
            self._cache_misses += 1
            
            # Check cache size and evict if needed
            if len(self._equity_cache) >= self._cache_max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._equity_cache))
                del self._equity_cache[oldest_key]
            
            # Calculate equity using Monte Carlo
            start_time = time.time()
            equity = self._monte_carlo_equity(hand, board, opponent_range, stage)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Check timeout
            if elapsed_ms > self.timeout_ms:
                # Fallback to estimate
                hand_strength = self.categorize_hand_strength(hand, board, stage)
                equity = self.estimate_equity(hand_strength['category'], opponent_range)
            else:
                # Cache result
                self._equity_cache[cache_key] = equity
            
            return equity
        except Exception as e:
            # Error handling - fallback to estimate
            try:
                hand_strength = self.categorize_hand_strength(hand, board, stage)
                return self.estimate_equity(hand_strength['category'], opponent_range)
            except Exception:
                return 50.0  # Default fallback
    
    def _monte_carlo_equity(self, hand, board, opponent_range, stage):
        """
        Calculate equity using Monte Carlo simulation.
        
        Args:
            hand (list): Player's hand
            board (list): Board cards
            opponent_range (dict): Opponent's range
            stage (int): Game stage
        
        Returns:
            float: Equity percentage (0-100)
        """
        if not opponent_range:
            # Default to tight range if no range provided
            opponent_range = self._get_default_opponent_range()
        
        # Get available cards (deck minus hand and board)
        used_cards = set(hand + board)
        available_cards = [i for i in range(52) if i not in used_cards]
        
        wins = 0
        losses = 0
        ties = 0
        
        # Determine how many cards to deal
        if stage == 1:  # Flop - need turn and river
            cards_to_deal = 2
        elif stage == 2:  # Turn - need river
            cards_to_deal = 1
        else:  # River - showdown
            cards_to_deal = 0
        
        iterations = min(self.monte_carlo_iterations, len(available_cards) // 2)
        
        for _ in range(iterations):
            # Sample opponent hand from range
            opponent_hand = self._sample_opponent_hand(opponent_range, used_cards)
            if not opponent_hand:
                continue
            
            # Deal remaining board cards
            remaining_cards = [c for c in available_cards if c not in opponent_hand]
            if len(remaining_cards) < cards_to_deal:
                continue
            
            if cards_to_deal > 0:
                board_complete = board + random.sample(remaining_cards, cards_to_deal)
            else:
                board_complete = board
            
            # Evaluate hands
            player_rank = self._evaluate_hand_rank(hand, board_complete)
            opponent_rank = self._evaluate_hand_rank(opponent_hand, board_complete)
            
            if player_rank > opponent_rank:
                wins += 1
            elif player_rank < opponent_rank:
                losses += 1
            else:
                ties += 1
        
        if iterations == 0:
            return 50.0  # Default if no iterations
        
        # Calculate equity
        equity = (wins + ties / 2) / iterations * 100
        return equity
    
    def _evaluate_hand_rank(self, hand, board):
        """
        Evaluate hand rank for comparison (higher is better).
        
        Returns integer rank: 9=straight flush, 8=quads, 7=full house, etc.
        """
        hand_cards = self._cards_to_suit_rank(hand)
        board_cards = self._cards_to_suit_rank(board)
        hand_eval = self._evaluate_hand(hand_cards, board_cards)
        return hand_eval['rank']
    
    def _parse_hand_string(self, hand_str):
        """
        Parse hand string (e.g., "AA", "AKs", "95o") to get rank requirements.
        
        Args:
            hand_str (str): Hand string like "AA", "AKs", "95o"
        
        Returns:
            dict: {
                'rank1': int,  # First rank (0=A, 1=2, ..., 12=K)
                'rank2': int,  # Second rank
                'is_pair': bool,
                'is_suited': bool,  # True for suited, False for offsuit, None for pairs
            }
        """
        rank_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        rank_map = {name: idx for idx, name in enumerate(rank_names)}
        
        hand_str = hand_str.upper().strip()
        
        # Check if pair (e.g., "AA", "KK")
        if len(hand_str) == 2 and hand_str[0] == hand_str[1]:
            rank = rank_map.get(hand_str[0])
            if rank is not None:
                return {
                    'rank1': rank,
                    'rank2': rank,
                    'is_pair': True,
                    'is_suited': None  # Pairs don't have suit requirement
                }
        
        # Check if suited (ends with 's')
        if hand_str.endswith('S'):
            ranks_str = hand_str[:-1]
            if len(ranks_str) == 2:
                rank1 = rank_map.get(ranks_str[0])
                rank2 = rank_map.get(ranks_str[1])
                if rank1 is not None and rank2 is not None:
                    # Ensure rank1 >= rank2
                    if rank1 < rank2:
                        rank1, rank2 = rank2, rank1
                    return {
                        'rank1': rank1,
                        'rank2': rank2,
                        'is_pair': False,
                        'is_suited': True
                    }
        
        # Check if offsuit (ends with 'o')
        if hand_str.endswith('O'):
            ranks_str = hand_str[:-1]
            if len(ranks_str) == 2:
                rank1 = rank_map.get(ranks_str[0])
                rank2 = rank_map.get(ranks_str[1])
                if rank1 is not None and rank2 is not None:
                    # Ensure rank1 >= rank2
                    if rank1 < rank2:
                        rank1, rank2 = rank2, rank1
                    return {
                        'rank1': rank1,
                        'rank2': rank2,
                        'is_pair': False,
                        'is_suited': False
                    }
        
        # Invalid hand string
        return None
    
    def _hand_string_to_card_indices(self, hand_str, used_cards):
        """
        Convert hand string to list of possible card index pairs.
        
        Args:
            hand_str (str): Hand string like "AA", "AKs", "95o"
            used_cards (set): Cards already used
        
        Returns:
            list: List of [card1, card2] pairs that match the hand string
        """
        hand_info = self._parse_hand_string(hand_str)
        if not hand_info:
            return []
        
        rank1 = hand_info['rank1']
        rank2 = hand_info['rank2']
        is_pair = hand_info['is_pair']
        is_suited = hand_info['is_suited']
        
        matching_hands = []
        
        if is_pair:
            # Pair: need two cards of same rank, different suits
            # Generate unique combinations (avoid duplicates like [card1, card2] and [card2, card1])
            for suit1 in range(4):
                for suit2 in range(suit1 + 1, 4):  # Only generate unique pairs
                    card1 = suit1 * 13 + rank1
                    card2 = suit2 * 13 + rank2
                    if card1 not in used_cards and card2 not in used_cards:
                        matching_hands.append([card1, card2])
        elif is_suited:
            # Suited: same suit, different ranks
            for suit in range(4):
                card1 = suit * 13 + rank1
                card2 = suit * 13 + rank2
                if card1 not in used_cards and card2 not in used_cards:
                    matching_hands.append([card1, card2])
        else:
            # Offsuit: different suits, different ranks
            for suit1 in range(4):
                for suit2 in range(4):
                    if suit1 != suit2:  # Different suits
                        card1 = suit1 * 13 + rank1
                        card2 = suit2 * 13 + rank2
                        if card1 not in used_cards and card2 not in used_cards:
                            matching_hands.append([card1, card2])
        
        return matching_hands
    
    def _sample_opponent_hand(self, opponent_range, used_cards):
        """
        Sample an opponent hand from their range, respecting frequencies.
        
        This method parses hand strings (e.g., "AA", "AKs", "95o") and converts them
        to card indices, then samples based on the frequencies in the range. Hands with
        higher frequencies are more likely to be selected.
        
        Args:
            opponent_range (dict): Opponent's range (hand_str -> probability)
                - hand_str: Hand string like "AA", "AKs", "95o"
                - probability: Frequency (0.0-1.0) indicating how often hand is in range
            used_cards (set): Cards already used (player's hand + board)
        
        Returns:
            list: Opponent hand as [card1, card2] card indices, or None if can't sample
        
        Example:
            >>> range_dict = {'AA': 1.0, '95o': 0.7}
            >>> used = {0, 1}  # Player's hand
            >>> hand = calculator._sample_opponent_hand(range_dict, used)
            >>> # AA will be sampled more often than 95o (1.0 vs 0.7 frequency)
        """
        if not opponent_range:
            # Fallback to random sampling if no range
            available = [i for i in range(52) if i not in used_cards]
            if len(available) < 2:
                return None
            return random.sample(available, 2)
        
        # Build weighted list of all possible hands from range
        weighted_hands = []
        for hand_str, frequency in opponent_range.items():
            # Skip if frequency is 0 or hand should be excluded
            if frequency <= 0:
                continue
            
            # Get all card combinations for this hand string
            possible_hands = self._hand_string_to_card_indices(hand_str, used_cards)
            
            # Add each possible hand with its frequency weight
            for hand in possible_hands:
                weighted_hands.append((hand, frequency))
        
        if not weighted_hands:
            # No valid hands in range, fallback to random
            available = [i for i in range(52) if i not in used_cards]
            if len(available) < 2:
                return None
            return random.sample(available, 2)
        
        # Weighted random selection based on frequencies
        # Hands with higher frequency are more likely to be selected
        weights = [freq for _, freq in weighted_hands]
        selected_hand = random.choices(
            [hand for hand, _ in weighted_hands],
            weights=weights,
            k=1
        )[0]
        
        return selected_hand
    
    def _get_default_opponent_range(self):
        """Get default wide opponent range using GTO button opening range."""
        # Use button opening range as widest reasonable default (~70% of hands)
        return self._get_default_wide_range()
    
    def _get_default_wide_range(self):
        """
        Get default wide range using GTO button opening range.
        
        Returns:
            dict: Opponent range (hand_str -> probability)
        """
        # Convert GTORules format to probability format
        # Supports both binary ('raise'/'call'/'fold') and frequency (0.0-1.0) formats
        gto_range = self.gto_rules.button_opening_range_100bb
        range_dict = {}
        
        for hand_str, action in gto_range.items():
            if isinstance(action, (int, float)):
                # Frequency format (0.0-1.0): hand raises with this frequency
                if 0.0 < action <= 1.0:
                    range_dict[hand_str] = float(action)
            elif action == 'raise':
                # Always raise
                range_dict[hand_str] = 1.0
            elif action == 'call':
                # Always call (include in range)
                range_dict[hand_str] = 1.0
            # 'fold' hands are excluded (not added to range_dict)
        
        return range_dict
    
    def _get_preflop_action(self, action_history, position):
        """
        Detect preflop action type from action history.
        
        Args:
            action_history (list): History of actions in the hand
            position (str): Opponent's position ('button' or 'big_blind')
        
        Returns:
            dict: {
                'action_type': str,  # 'open', 'call', '3bet', '4bet', 'all_in'
                'position': str,
                'stack_depth': float,
                'bet_size': float  # Bet size in BB if available
            }
        """
        if not action_history:
            return {
                'action_type': 'open' if position == 'button' else 'call',
                'position': position,
                'stack_depth': 100.0,  # Default
                'bet_size': 0.0
            }
        
        # Find first preflop action (stage == 0 or first action)
        first_action = None
        for action in action_history:
            if isinstance(action, dict):
                stage = action.get('stage', 0)
                if isinstance(stage, str):
                    # Handle string stages
                    if stage.lower() in ('preflop', '0'):
                        stage = 0
                    else:
                        continue
                
                if stage == 0:
                    first_action = action
                    break
            elif isinstance(action, (int, str)):
                # Legacy format - assume preflop if it's the first action
                if first_action is None:
                    first_action = {'action': action, 'stage': 0}
                    break
        
        if not first_action:
            # No preflop action found, use default
            return {
                'action_type': 'open' if position == 'button' else 'call',
                'position': position,
                'stack_depth': 100.0,
                'bet_size': 0.0
            }
        
        # Extract action string
        action_str = first_action.get('action', '')
        if isinstance(action_str, int):
            # Legacy integer format: 0=Fold, 1=Check/Call, 2=Raise ½ Pot, 3=Raise Pot, 4=All-In
            if action_str == 1:
                action_str = 'call'
            elif action_str == 2:
                action_str = 'raise'
            elif action_str == 3:
                action_str = 'raise'
            elif action_str == 4:
                action_str = 'all_in'
            else:
                action_str = 'fold'
        else:
            action_str = str(action_str).lower()
        
        # Extract bet size if available
        bet_size = first_action.get('bet_amount', 0)
        pot = first_action.get('pot', 0)
        big_blind = first_action.get('big_blind', 2)
        
        # Detect action type from label
        action_type = None
        bet_size_bb = 0.0
        
        # Handle new action label formats
        if 'raise to' in action_str or '3-bet to' in action_str or '4-bet to' in action_str:
            # Extract BB amount from labels like "Raise to 3BB", "3-bet to 7BB", "4-bet to 25BB"
            import re
            match = re.search(r'(\d+(?:\.\d+)?)\s*bb', action_str, re.IGNORECASE)
            if match:
                bet_size_bb = float(match.group(1))
            
            if '4-bet' in action_str or '4bet' in action_str:
                action_type = '4bet'
            elif '3-bet' in action_str or '3bet' in action_str:
                action_type = '3bet'
            else:
                # Regular raise - determine if it's an open or 3-bet based on pot size
                if pot > 0 and big_blind > 0:
                    pot_bb = pot / big_blind
                    if pot_bb < 5:
                        action_type = 'open'
                    elif pot_bb < 12:
                        action_type = '3bet'
                    else:
                        action_type = '4bet'
                else:
                    action_type = 'open'
        elif 'all-in' in action_str or 'all in' in action_str:
            action_type = 'all_in'
            if bet_size > 0 and big_blind > 0:
                bet_size_bb = bet_size / big_blind
        elif 'call' in action_str or 'check' in action_str:
            action_type = 'call'
        elif 'fold' in action_str:
            action_type = 'fold'
        elif 'raise' in action_str or 'bet' in action_str:
            # Legacy formats: "Raise ½ Pot", "Raise Pot", "Bet ½ Pot"
            if pot > 0 and big_blind > 0:
                pot_bb = pot / big_blind
                if pot_bb < 5:
                    action_type = 'open'
                elif pot_bb < 12:
                    action_type = '3bet'
                else:
                    action_type = '4bet'
            else:
                action_type = 'open'
        else:
            # Default based on position
            action_type = 'open' if position == 'button' else 'call'
        
        # Calculate bet size if not already extracted
        if bet_size_bb == 0.0 and bet_size > 0 and big_blind > 0:
            bet_size_bb = bet_size / big_blind
        
        return {
            'action_type': action_type,
            'position': position,
            'stack_depth': 100.0,  # Will be adjusted by caller
            'bet_size': bet_size_bb
        }
    
    def _get_preflop_range_from_action(self, action_history, position, stack_depth):
        """
        Select appropriate GTO preflop range based on action and position.
        
        Args:
            action_history (list): History of actions in the hand
            position (str): Opponent's position ('button' or 'big_blind')
            stack_depth (float): Stack depth in big blinds
        
        Returns:
            dict: Opponent range (hand_str -> probability)
        """
        # Get preflop action info
        action_info = self._get_preflop_action(action_history, position)
        action_type = action_info['action_type']
        
        # Select appropriate GTO range based on stack depth
        if stack_depth <= 30:
            # Short stack (20-30 BB)
            if position == 'button':
                if action_type == 'open':
                    gto_range = getattr(self.gto_rules, 'button_opening_range_20_30bb', 
                                      self.gto_rules.button_opening_range_100bb)
                elif action_type == '3bet':
                    gto_range = getattr(self.gto_rules, 'button_3bet_range_20_30bb',
                                      self.gto_rules.button_3bet_range_100bb)
                elif action_type == '4bet':
                    gto_range = getattr(self.gto_rules, 'button_4bet_range_20_30bb',
                                      self.gto_rules.button_4bet_range_100bb)
                else:
                    gto_range = self.gto_rules.button_opening_range_100bb
            else:  # big_blind
                if action_type == 'call':
                    gto_range = getattr(self.gto_rules, 'bb_defending_range_20_30bb',
                                      self.gto_rules.bb_defending_range_100bb)
                elif action_type == '3bet':
                    gto_range = getattr(self.gto_rules, 'bb_3bet_range_20_30bb',
                                      self.gto_rules.bb_3bet_range_100bb)
                elif action_type == '4bet':
                    gto_range = getattr(self.gto_rules, 'bb_4bet_range_20_30bb',
                                      self.gto_rules.bb_4bet_range_100bb)
                else:
                    gto_range = self.gto_rules.bb_defending_range_100bb
        elif stack_depth <= 50:
            # Medium stack (50 BB)
            if position == 'button':
                if action_type == 'open':
                    gto_range = self.gto_rules.button_opening_range_50bb
                elif action_type == '3bet':
                    gto_range = getattr(self.gto_rules, 'button_3bet_range_50bb',
                                      self.gto_rules.button_3bet_range_100bb)
                elif action_type == '4bet':
                    gto_range = getattr(self.gto_rules, 'button_4bet_range_50bb',
                                      self.gto_rules.button_4bet_range_100bb)
                else:
                    gto_range = self.gto_rules.button_opening_range_50bb
            else:  # big_blind
                if action_type == 'call':
                    gto_range = getattr(self.gto_rules, 'bb_defending_range_50bb',
                                      self.gto_rules.bb_defending_range_100bb)
                elif action_type == '3bet':
                    gto_range = getattr(self.gto_rules, 'bb_3bet_range_50bb',
                                      self.gto_rules.bb_3bet_range_100bb)
                elif action_type == '4bet':
                    gto_range = getattr(self.gto_rules, 'bb_4bet_range_50bb',
                                      self.gto_rules.bb_4bet_range_100bb)
                else:
                    gto_range = self.gto_rules.bb_defending_range_100bb
        else:
            # Deep stack (100+ BB)
            if position == 'button':
                if action_type == 'open':
                    gto_range = self.gto_rules.button_opening_range_100bb
                elif action_type == '3bet':
                    gto_range = self.gto_rules.button_3bet_range_100bb
                elif action_type == '4bet':
                    gto_range = self.gto_rules.button_4bet_range_100bb
                else:
                    gto_range = self.gto_rules.button_opening_range_100bb
            else:  # big_blind
                if action_type == 'call':
                    gto_range = self.gto_rules.bb_defending_range_100bb
                elif action_type == '3bet':
                    gto_range = self.gto_rules.bb_3bet_range_100bb
                elif action_type == '4bet':
                    gto_range = self.gto_rules.bb_4bet_range_100bb
                else:
                    gto_range = self.gto_rules.bb_defending_range_100bb
        
        # Handle all-in scenarios (wider ranges for short stacks)
        if action_info['action_type'] == 'all_in':
            # For all-in with short stacks, use wider opening ranges
            if stack_depth <= 30:
                if position == 'button':
                    gto_range = self.gto_rules.button_opening_range_20_30bb
                else:
                    gto_range = self.gto_rules.bb_defending_range_20_30bb
        
        # Convert GTORules format to probability format
        # Supports both binary ('raise'/'call'/'fold') and frequency (0.0-1.0) formats
        range_dict = {}
        for hand_str, action in gto_range.items():
            if isinstance(action, (int, float)):
                # Frequency format (0.0-1.0): hand raises with this frequency
                if 0.0 < action <= 1.0:
                    range_dict[hand_str] = float(action)
            elif action == 'raise':
                # Always raise
                range_dict[hand_str] = 1.0
            elif action == 'call':
                # Always call (include in range)
                range_dict[hand_str] = 1.0
            # 'fold' hands are excluded (not added to range_dict)
        
        return range_dict
    
    def _construct_opponent_range(self, action_history, position, stack_depth, current_stage=0, board=None):
        """
        Construct opponent range based on action history, position, and stack depth.
        
        Uses a hybrid approach combining GTO range templates with action-based narrowing:
        1. Preflop: Selects appropriate GTO range based on first action (open, call, 3-bet, 4-bet)
        2. Postflop: Narrows preflop range based on actions, bet sizing, and board texture
        
        Args:
            action_history (list): History of actions in the hand. Each action should be a dict with:
                - 'action': str (e.g., "Raise to 3BB", "Bet ½ Pot", "Call")
                - 'stage': int (0=preflop, 1=flop, 2=turn, 3=river)
                - 'bet_amount': float (optional, bet amount in chips)
                - 'pot': float (optional, pot size)
                - 'big_blind': float (optional, big blind size)
            position (str): Opponent's position ('button' or 'big_blind')
            stack_depth (float): Stack depth in big blinds
            current_stage (int): Current game stage (0=preflop, 1=flop, 2=turn, 3=river)
            board (list, optional): Board cards as RLCard card indices (for postflop narrowing)
        
        Returns:
            dict: Opponent range (hand_str -> probability), where hand_str is like "AA", "AKs", "AKo"
        
        Example:
            >>> action_history = [
            ...     {'action': 'Raise to 3BB', 'stage': 0, 'pot': 3, 'big_blind': 2},
            ...     {'action': 'Call', 'stage': 0}
            ... ]
            >>> range_dict = calculator._construct_opponent_range(
            ...     action_history, 'button', 100, current_stage=0
            ... )
            >>> len(range_dict)  # Should be ~47 hands for button opening range
            47
        """
        # Handle preflop (current_stage == 0)
        if current_stage == 0:
            # Use preflop range construction
            if action_history:
                return self._get_preflop_range_from_action(action_history, position, stack_depth)
            else:
                # No action history, use default wide range
                return self._get_default_opponent_range()
        
        # Handle postflop (current_stage > 0)
        # First get preflop range
        preflop_range = self._get_preflop_range_from_action(action_history, position, stack_depth)
        
        # Then narrow based on postflop actions
        if board:
            narrowed_range = self._narrow_range_postflop(
                preflop_range, action_history, current_stage, board
            )
            # Adjust for board texture
            board_texture = self._analyze_board_texture(board)
            final_range = self._adjust_range_for_board(
                narrowed_range, board, board_texture
            )
            return final_range
        
        return preflop_range
    
    def _analyze_postflop_actions(self, action_history, current_stage):
        """
        Analyze postflop actions to determine range narrowing strategy.
        
        Args:
            action_history (list): History of actions in the hand
            current_stage (int): Current game stage (1=flop, 2=turn, 3=river)
        
        Returns:
            dict: {
                'actions': list,  # List of action types
                'bet_sizes': list,  # List of bet amounts
                'bet_fractions': list,  # List of bet fractions (bet/pot)
                'action_sequence': list  # Sequence of who acted and when
            }
        """
        postflop_actions = []
        bet_sizes = []
        bet_fractions = []
        action_sequence = []
        
        for action in action_history:
            if isinstance(action, dict):
                stage = action.get('stage', 0)
                if isinstance(stage, str):
                    # Handle string stages
                    if stage.lower() in ('flop', '1'):
                        stage = 1
                    elif stage.lower() in ('turn', '2'):
                        stage = 2
                    elif stage.lower() in ('river', '3'):
                        stage = 3
                    else:
                        continue
                
                # Only process postflop actions (stage > 0 and <= current_stage)
                if stage > 0 and stage <= current_stage:
                    action_str = action.get('action', '')
                    if isinstance(action_str, int):
                        # Legacy integer format
                        if action_str == 1:
                            action_str = 'call'
                        elif action_str == 2:
                            action_str = 'bet'
                        elif action_str == 3:
                            action_str = 'bet'
                        elif action_str == 4:
                            action_str = 'all_in'
                        else:
                            action_str = 'fold'
                    else:
                        action_str = str(action_str).lower()
                    
                    # Extract bet sizing
                    bet_amount = action.get('bet_amount', 0)
                    pot = action.get('pot', 0)
                    big_blind = action.get('big_blind', 2)
                    bet_fraction = 0.0
                    
                    # Handle new action label formats
                    if 'bet' in action_str or 'raise' in action_str:
                        # Extract bet sizing from labels
                        if '½ pot' in action_str or '1/2 pot' in action_str or '0.5 pot' in action_str:
                            bet_fraction = 0.5
                        elif '⅔ pot' in action_str or '2/3 pot' in action_str or '0.67 pot' in action_str:
                            bet_fraction = 0.67
                        elif 'pot' in action_str and 'bet' in action_str:
                            bet_fraction = 1.0
                        elif 'raise to' in action_str:
                            # Extract BB amount from "Raise to X BB"
                            import re
                            match = re.search(r'(\d+(?:\.\d+)?)\s*bb', action_str, re.IGNORECASE)
                            if match and pot > 0:
                                bet_size_bb = float(match.group(1))
                                bet_amount = bet_size_bb * big_blind
                                bet_fraction = bet_amount / pot if pot > 0 else 0.0
                        elif bet_amount > 0 and pot > 0:
                            bet_fraction = bet_amount / pot
                    
                    postflop_actions.append(action_str)
                    bet_sizes.append(bet_amount)
                    bet_fractions.append(bet_fraction)
                    action_sequence.append({
                        'stage': stage,
                        'action': action_str,
                        'bet_amount': bet_amount,
                        'bet_fraction': bet_fraction
                    })
        
        return {
            'actions': postflop_actions,
            'bet_sizes': bet_sizes,
            'bet_fractions': bet_fractions,
            'action_sequence': action_sequence
        }
    
    def _narrow_range_postflop(self, preflop_range, action_history, current_stage, board):
        """
        Narrow opponent range based on postflop actions.
        
        Args:
            preflop_range (dict): Preflop range (hand_str -> probability)
            action_history (list): History of actions in the hand
            current_stage (int): Current game stage (1=flop, 2=turn, 3=river)
            board (list): Board cards
        
        Returns:
            dict: Narrowed range (hand_str -> probability)
        """
        # Analyze postflop actions
        action_analysis = self._analyze_postflop_actions(action_history, current_stage)
        
        # Start with preflop range
        narrowed_range = preflop_range.copy()
        
        # Apply narrowing based on actions and bet sizing
        for action_info in action_analysis['action_sequence']:
            action_type = action_info['action']
            bet_fraction = action_info['bet_fraction']
            bet_amount = action_info.get('bet_amount', 0)
            
            if 'bet' in action_type or 'raise' in action_type:
                # Betting narrows range - remove weak hands
                # Use bet sizing analysis to determine narrowing amount
                sizing_analysis = self._analyze_bet_sizing(bet_fraction, is_preflop=False)
                narrowed_range = self._remove_weak_hands(
                    narrowed_range, bet_fraction, board, current_stage
                )
            elif 'call' in action_type:
                # Calling removes very strong hands (would bet for value)
                narrowed_range = self._remove_strong_hands(
                    narrowed_range, board, current_stage
                )
            elif 'check' in action_type:
                # Checking keeps wider range (could be strong or weak)
                pass
            elif 'fold' in action_type:
                # Folding removes all hands (shouldn't happen for opponent)
                return {}
        
        return narrowed_range
    
    def _analyze_bet_sizing(self, bet_fraction, is_preflop=False, bet_size_bb=0.0):
        """
        Analyze bet sizing to infer hand strength and polarization.
        
        Args:
            bet_fraction (float): Bet size as fraction of pot
            is_preflop (bool): Whether this is a preflop bet
            bet_size_bb (float): Bet size in big blinds (for preflop)
        
        Returns:
            dict: {
                'strength': str,  # 'very_strong', 'strong', 'medium', 'weak'
                'polarization': str,  # 'high', 'medium', 'low'
                'likely_hands': str  # Description of likely hand range
            }
        """
        if is_preflop:
            # Preflop sizing analysis
            if bet_size_bb >= 20:
                return {
                    'strength': 'very_strong',
                    'polarization': 'high',
                    'likely_hands': 'premium_value_or_all_in'
                }
            elif bet_size_bb >= 15:
                return {
                    'strength': 'very_strong',
                    'polarization': 'medium',
                    'likely_hands': 'premium_value'
                }
            elif bet_size_bb >= 7:
                return {
                    'strength': 'strong',
                    'polarization': 'low',
                    'likely_hands': 'value_or_bluff'
                }
            elif bet_size_bb >= 3:
                return {
                    'strength': 'medium',
                    'polarization': 'low',
                    'likely_hands': 'balanced_range'
                }
            else:
                return {
                    'strength': 'weak',
                    'polarization': 'low',
                    'likely_hands': 'weak_value_or_bluff'
                }
        else:
            # Postflop sizing analysis
            if bet_fraction >= 1.5:
                return {
                    'strength': 'very_strong',
                    'polarization': 'high',
                    'likely_hands': 'polarized_nuts_or_air'
                }
            elif bet_fraction >= 0.75:
                return {
                    'strength': 'strong',
                    'polarization': 'medium',
                    'likely_hands': 'strong_value_or_strong_bluff'
                }
            elif bet_fraction >= 0.5:
                return {
                    'strength': 'medium',
                    'polarization': 'low',
                    'likely_hands': 'balanced_range'
                }
            else:
                return {
                    'strength': 'weak',
                    'polarization': 'low',
                    'likely_hands': 'weak_value_or_weak_bluff'
                }
    
    def _remove_weak_hands(self, range_dict, bet_fraction, board, stage):
        """
        Remove weak hands that would check/fold when opponent bets.
        
        Args:
            range_dict (dict): Current range
            bet_fraction (float): Bet size as fraction of pot
            board (list): Board cards
            stage (int): Current stage
        
        Returns:
            dict: Range with weak hands removed
        """
        # Analyze bet sizing to determine how much to narrow
        sizing_analysis = self._analyze_bet_sizing(bet_fraction, is_preflop=False)
        
        # Adjust narrowing based on bet sizing
        if sizing_analysis['polarization'] == 'high':
            # Polarized bet - remove middle-strength hands, keep very strong and very weak
            # For now, keep top 40% and bottom 20% (simplified)
            return self._keep_top_percentage(range_dict, 0.6)
        elif sizing_analysis['strength'] == 'very_strong' or sizing_analysis['strength'] == 'strong':
            # Strong bet - remove more weak hands
            return self._keep_top_percentage(range_dict, 0.65)
        elif sizing_analysis['strength'] == 'medium':
            # Medium bet - remove bottom 30% of range
            return self._keep_top_percentage(range_dict, 0.7)
        else:
            # Small bet - remove bottom 20% of range
            return self._keep_top_percentage(range_dict, 0.8)
    
    def _remove_strong_hands(self, range_dict, board, stage):
        """
        Remove very strong hands that would bet for value when opponent calls.
        
        Args:
            range_dict (dict): Current range
            board (list): Board cards
            stage (int): Current stage
        
        Returns:
            dict: Range with strong hands removed
        """
        # When opponent calls, they likely don't have the nuts
        # Remove top 10% of range (very strong hands that would bet)
        return self._keep_top_percentage(range_dict, 0.9)
    
    def _keep_top_percentage(self, range_dict, percentage):
        """
        Keep top percentage of range based on hand strength.
        
        This is a simplified implementation. In a full implementation,
        we'd rank hands by strength and keep the top N%.
        
        Args:
            range_dict (dict): Current range
            percentage (float): Percentage to keep (0.0 to 1.0)
        
        Returns:
            dict: Filtered range
        """
        # Simplified: just return a subset
        # In a full implementation, we'd rank hands by strength
        total_hands = len(range_dict)
        hands_to_keep = max(1, int(total_hands * percentage))
        
        # For now, return all hands (simplified)
        # TODO: Implement proper hand strength ranking
        return range_dict
    
    def _adjust_range_for_board(self, range_dict, board, board_texture):
        """
        Adjust range based on how board hits opponent's range.
        
        Args:
            range_dict (dict): Current range
            board (list): Board cards
            board_texture (dict): Board texture analysis result
        
        Returns:
            dict: Adjusted range
        """
        adjusted_range = {}
        
        # Filter hands that make sense on this board
        for hand_str, prob in range_dict.items():
            if self._hand_makes_sense_on_board(hand_str, board, board_texture):
                # Adjust probability based on board texture
                adjusted_prob = self._calculate_hand_probability_on_board(
                    hand_str, board, board_texture, prob
                )
                if adjusted_prob > 0:
                    adjusted_range[hand_str] = adjusted_prob
        
        return adjusted_range
    
    def _hand_makes_sense_on_board(self, hand_str, board, board_texture):
        """
        Check if hand makes sense on this board given actions.
        
        Args:
            hand_str (str): Hand string (e.g., "AA", "AKs")
            board (list): Board cards
            board_texture (dict): Board texture analysis
        
        Returns:
            bool: True if hand makes sense on board
        """
        # Simplified: for now, all hands make sense
        # In a full implementation, we'd check if hand conflicts with board
        # (e.g., if board has A♠ and hand is "AKs", check if it's A♠K♠)
        return True
    
    def _calculate_hand_probability_on_board(self, hand_str, board, board_texture, base_prob):
        """
        Calculate hand probability on board based on texture.
        
        Args:
            hand_str (str): Hand string
            board (list): Board cards
            board_texture (dict): Board texture analysis
            base_prob (float): Base probability
        
        Returns:
            float: Adjusted probability
        """
        # Simplified: return base probability
        # In a full implementation, we'd adjust based on:
        # - Wet boards: Increase probability for draws
        # - Dry boards: Keep polarized ranges
        # - Paired boards: Adjust for hands that make sense
        
        board_type = board_texture.get('type', 'unknown')
        
        # Adjust based on board texture
        if board_type == 'wet':
            # Wet boards: Opponent's range may include more draws
            # Keep probability similar
            return base_prob
        elif board_type == 'dry':
            # Dry boards: More polarized range
            # Keep probability similar
            return base_prob
        else:
            # Default: keep base probability
            return base_prob
    
    def _analyze_board_texture(self, board):
        """
        Analyze board texture.
        
        Args:
            board (list): Board cards
        
        Returns:
            dict: {
                'type': str,  # 'wet', 'dry', 'dynamic', 'static'
                'draws': list,  # List of available draws
                'coordination': str  # 'high', 'medium', 'low'
            }
        """
        if not board or len(board) < 3:
            return {'type': 'unknown', 'draws': [], 'coordination': 'unknown'}
        
        board_cards = self._cards_to_suit_rank(board)
        
        # Check for draws
        draws = []
        if self._has_flush_draw(board_cards):
            draws.append('flush')
        if self._has_straight_draw(board_cards):
            draws.append('straight')
        
        # Count suits and ranks
        suit_counts = {}
        rank_counts = {}
        for suit, rank in board_cards:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Determine coordination
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        max_rank_count = max(rank_counts.values()) if rank_counts else 0
        
        if max_suit_count >= 3 or max_rank_count >= 2 or len(draws) > 0:
            coordination = 'high'
        elif max_suit_count >= 2:
            coordination = 'medium'
        else:
            coordination = 'low'
        
        # Determine type
        if len(draws) >= 2 or coordination == 'high':
            board_type = 'wet'
        elif len(draws) == 0 and coordination == 'low':
            board_type = 'dry'
        elif len(draws) > 0:
            board_type = 'dynamic'
        else:
            board_type = 'static'
        
        return {
            'type': board_type,
            'draws': draws,
            'coordination': coordination
        }
    
    def _cards_to_suit_rank(self, cards):
        """
        Convert RLCard card indices to (suit, rank) tuples.
        
        RLCard uses 0-51: suit = card // 13, rank = card % 13
        Rank: 0=A, 1=2, 2=3, ..., 12=K
        
        Args:
            cards (list): List of card indices
        
        Returns:
            list: List of (suit, rank) tuples
        """
        result = []
        for card in cards:
            # Convert to int if it's a string or numpy type
            if not isinstance(card, int):
                try:
                    card = int(card)
                except (ValueError, TypeError):
                    card = 0  # Safe fallback
            suit = card // 13
            rank = card % 13
            result.append((suit, rank))
        return result
    
    def _evaluate_hand(self, hand_cards, board_cards):
        """
        Evaluate hand strength (simplified poker hand evaluation).
        
        Returns dict with rank and description.
        Hand ranks: 9=straight flush, 8=quads, 7=full house, 6=flush,
                    5=straight, 4=trips, 3=two pair, 2=one pair, 1=high card
        """
        all_cards = hand_cards + board_cards
        
        # Count ranks and suits
        rank_counts = {}
        suit_counts = {}
        for suit, rank in all_cards:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        # Check for straight flush (simplified - just check flush and straight separately)
        # Check for quads
        if 4 in rank_counts.values():
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            return {'rank': 8, 'description': f"Four of a kind {self._rank_to_name(quad_rank)}"}
        
        # Check for full house
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trips_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return {'rank': 7, 'description': f"Full house {self._rank_to_name(trips_rank)}s over {self._rank_to_name(pair_rank)}s"}
        
        # Check for flush
        if max(suit_counts.values()) >= 5:
            flush_suit = [s for s, c in suit_counts.items() if c >= 5][0]
            return {'rank': 6, 'description': f"Flush"}
        
        # Check for straight (simplified)
        ranks = sorted(set([r for r in rank_counts.keys()]))
        if len(ranks) >= 5:
            # Check for straight
            for i in range(len(ranks) - 4):
                if ranks[i+4] - ranks[i] == 4:
                    return {'rank': 5, 'description': f"Straight"}
        
        # Check for trips
        if 3 in rank_counts.values():
            trips_rank = [r for r, c in rank_counts.items() if c == 3][0]
            return {'rank': 4, 'description': f"Three of a kind {self._rank_to_name(trips_rank)}"}
        
        # Check for two pair
        pairs = [r for r, c in rank_counts.items() if c == 2]
        if len(pairs) >= 2:
            pairs.sort(reverse=True)
            return {'rank': 3, 'description': f"Two pair {self._rank_to_name(pairs[0])}s and {self._rank_to_name(pairs[1])}s"}
        
        # Check for one pair
        if 2 in rank_counts.values():
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return {'rank': 2, 'pair_rank': pair_rank, 'description': f"Pair of {self._rank_to_name(pair_rank)}s"}
        
        # High card
        high_rank = max([r for r in rank_counts.keys()])
        return {'rank': 1, 'description': f"High card {self._rank_to_name(high_rank)}"}
    
    def _has_draw(self, all_cards):
        """Check if hand has a draw (flush or straight draw)."""
        return self._has_flush_draw(all_cards) or self._has_straight_draw(all_cards)
    
    def _has_flush_draw(self, all_cards):
        """Check for flush draw (4 cards of same suit)."""
        suit_counts = {}
        for suit, rank in all_cards:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        return max(suit_counts.values()) >= 4
    
    def _has_straight_draw(self, all_cards):
        """Check for straight draw (4 cards in sequence)."""
        ranks = sorted(set([r for r, _ in all_cards]))
        if len(ranks) >= 4:
            for i in range(len(ranks) - 3):
                if ranks[i+3] - ranks[i] <= 4:
                    return True
        return False
    
    def _get_draw_type(self, all_cards):
        """Get type of draw."""
        has_flush = self._has_flush_draw(all_cards)
        has_straight = self._has_straight_draw(all_cards)
        
        if has_flush and has_straight:
            return "Combo draw (flush and straight)"
        elif has_flush:
            return "Flush draw"
        elif has_straight:
            return "Straight draw"
        else:
            return "Draw"
    
    def _rank_to_name(self, rank):
        """Convert rank number to card name."""
        names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        return names[rank] if 0 <= rank < 13 else '?'

