"""
Unit tests for EquityCalculator Phase 2 enhancements
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from coach.equity_calculator import EquityCalculator
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from webapp.coach.equity_calculator import EquityCalculator


class TestEquityCalculatorPhase2(unittest.TestCase):
    """Test EquityCalculator Phase 2 enhancements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = EquityCalculator(cache_size=100, monte_carlo_iterations=100, timeout_ms=500)
    
    def test_board_texture_analysis(self):
        """Test board texture analysis."""
        # Wet board (flush draw + straight draw)
        board = [0, 1, 2, 13, 14]  # Simplified - would need actual card indices
        texture = self.calculator._analyze_board_texture(board)
        self.assertIn('type', texture)
        self.assertIn('draws', texture)
        self.assertIn('coordination', texture)
    
    def test_opponent_range_construction(self):
        """Test opponent range construction."""
        action_history = [
            {'player_id': 1, 'action': 'Raise to 3BB', 'stage': 0, 'pot': 3, 'big_blind': 2},
            {'player_id': 0, 'action': 'Call', 'stage': 0}
        ]
        range_dict = self.calculator._construct_opponent_range(
            action_history, 'button', 100, current_stage=0
        )
        self.assertIsInstance(range_dict, dict)
        self.assertGreater(len(range_dict), 0)
        # Should have wider range than old default (8 hands)
        self.assertGreater(len(range_dict), 8)
    
    def test_preflop_range_selection(self):
        """Test preflop range selection for different actions."""
        # Test button opening range
        action_history = [
            {'player_id': 1, 'action': 'Raise to 3BB', 'stage': 0, 'pot': 3, 'big_blind': 2}
        ]
        range_dict = self.calculator._construct_opponent_range(
            action_history, 'button', 100, current_stage=0
        )
        self.assertGreater(len(range_dict), 40)  # Button opening range should be wide (47 hands)
        
        # Test BB defending range
        action_history = [
            {'player_id': 0, 'action': 'Raise to 3BB', 'stage': 0, 'pot': 3, 'big_blind': 2},
            {'player_id': 1, 'action': 'Call', 'stage': 0}
        ]
        range_dict = self.calculator._construct_opponent_range(
            action_history, 'big_blind', 100, current_stage=0
        )
        self.assertGreater(len(range_dict), 30)  # BB defending range should be wide
    
    def test_default_range_is_wide(self):
        """Test that default range uses GTO ranges instead of tight default."""
        # No action history - should use default wide range
        range_dict = self.calculator._get_default_opponent_range()
        self.assertIsInstance(range_dict, dict)
        # Should have much wider range than old 8-hand default (47 hands vs 8)
        self.assertGreater(len(range_dict), 40)
    
    def test_mixed_strategies_in_range(self):
        """Test that ranges include mixed strategies (frequencies < 1.0)."""
        range_dict = self.calculator._get_default_opponent_range()
        
        # Check that 95o has a frequency (should be ~0.7)
        self.assertIn('95o', range_dict)
        self.assertIsInstance(range_dict['95o'], (int, float))
        self.assertGreater(range_dict['95o'], 0.0)
        self.assertLessEqual(range_dict['95o'], 1.0)
        # Should be around 0.7 (70% raise frequency)
        self.assertGreaterEqual(range_dict['95o'], 0.5)
        self.assertLessEqual(range_dict['95o'], 0.8)
        
        # Check that premium hands have frequency 1.0 (always raise)
        self.assertEqual(range_dict.get('AA', 0), 1.0)
        self.assertEqual(range_dict.get('KK', 0), 1.0)
        
        # Check that some hands have mixed strategies
        mixed_strategy_hands = [h for h, f in range_dict.items() if 0.0 < f < 1.0]
        self.assertGreater(len(mixed_strategy_hands), 0, "Should have some hands with mixed strategies")
    
    def test_hand_string_parsing(self):
        """Test hand string parsing to card indices."""
        # Test pair
        hands_aa = self.calculator._hand_string_to_card_indices('AA', set())
        self.assertEqual(len(hands_aa), 6)  # 6 ways to make AA (4 choose 2)
        
        # Test suited
        hands_aks = self.calculator._hand_string_to_card_indices('AKs', set())
        self.assertEqual(len(hands_aks), 4)  # 4 suits
        
        # Test offsuit
        hands_95o = self.calculator._hand_string_to_card_indices('95o', set())
        self.assertEqual(len(hands_95o), 12)  # 4*3 = 12 ways (4 suits for first, 3 for second)
        
        # Test with used cards
        used = {0, 13}  # Use two aces
        hands_aa_used = self.calculator._hand_string_to_card_indices('AA', used)
        self.assertEqual(len(hands_aa_used), 1)  # Only one AA left (26, 39)
    
    def test_frequency_based_sampling(self):
        """Test that hand sampling respects frequencies."""
        # Create a simple range with known frequencies
        test_range = {
            'AA': 1.0,  # Always in range
            '95o': 0.5,  # 50% frequency
            '72o': 0.0,  # Never in range (should be excluded)
        }
        
        used = set()
        samples = []
        for _ in range(100):
            hand = self.calculator._sample_opponent_hand(test_range, used)
            if hand:
                samples.append(hand)
        
        # Should get some samples
        self.assertGreater(len(samples), 0)
        
        # Verify 72o is never sampled (frequency 0.0)
        # This is hard to test directly, but we can verify AA and 95o can be sampled
        # by checking that we get valid hands
        self.assertTrue(all(isinstance(h, list) and len(h) == 2 for h in samples))
    
    def test_equity_calculation_caching(self):
        """Test equity calculation caching."""
        hand = [0, 1]  # Simplified
        board = [2, 3, 4]
        opponent_range = {'AA': 1.0, 'KK': 1.0}
        
        # First call - should miss cache
        equity1 = self.calculator.calculate_full_equity(hand, board, opponent_range, 1)
        
        # Second call - should hit cache
        equity2 = self.calculator.calculate_full_equity(hand, board, opponent_range, 1)
        
        # Should be same result
        self.assertEqual(equity1, equity2)
        
        # Check cache stats
        self.assertGreater(self.calculator._cache_hits, 0)
    
    def test_equity_calculation_timeout(self):
        """Test equity calculation timeout handling."""
        # Use very short timeout to test timeout handling
        calculator = EquityCalculator(timeout_ms=1)
        hand = [0, 1]
        board = [2, 3, 4]
        opponent_range = {'AA': 1.0}
        
        # Should fallback to estimate on timeout
        equity = calculator.calculate_full_equity(hand, board, opponent_range, 1)
        self.assertIsNotNone(equity)
        self.assertGreaterEqual(equity, 0)
        self.assertLessEqual(equity, 100)
    
    def test_error_handling_invalid_board(self):
        """Test error handling for invalid board states."""
        hand = [0, 1]
        
        # Invalid board (wrong number of cards)
        invalid_board = [2, 3]  # Only 2 cards
        equity = self.calculator.calculate_full_equity(hand, invalid_board, {}, 1)
        self.assertIsNone(equity)  # Should return None for invalid board
        
        # Duplicate cards
        duplicate_board = [2, 2, 3, 4, 5]  # Duplicate card
        equity = self.calculator.calculate_full_equity(hand, duplicate_board, {}, 1)
        self.assertIsNone(equity)  # Should return None for duplicate cards


if __name__ == '__main__':
    unittest.main()

