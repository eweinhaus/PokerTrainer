"""
Unit tests for GTORules Phase 2 enhancements
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from coach.gto_rules import GTORules
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from webapp.coach.gto_rules import GTORules


class TestGTORulesPhase2(unittest.TestCase):
    """Test GTORules Phase 2 enhancements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gto_rules = GTORules()
    
    def test_stack_depth_categorization(self):
        """Test stack depth categorization."""
        # Test 20-30 BB category
        self.assertEqual(self.gto_rules._get_stack_depth_category(20), '20_30')
        self.assertEqual(self.gto_rules._get_stack_depth_category(25), '20_30')
        self.assertEqual(self.gto_rules._get_stack_depth_category(29), '20_30')
        
        # Test 50 BB category
        self.assertEqual(self.gto_rules._get_stack_depth_category(30), '50')
        self.assertEqual(self.gto_rules._get_stack_depth_category(50), '50')
        self.assertEqual(self.gto_rules._get_stack_depth_category(99), '50')
        
        # Test 100+ BB category
        self.assertEqual(self.gto_rules._get_stack_depth_category(100), '100')
        self.assertEqual(self.gto_rules._get_stack_depth_category(200), '100')
    
    def test_button_opening_ranges_all_stack_depths(self):
        """Test button opening ranges for all stack depths."""
        # Test 20-30 BB (wider range)
        action = self.gto_rules.get_preflop_action('A9o', 'button', 'open', 25)
        self.assertEqual(action, 'raise')  # Should be in wider range
        
        # Test 50 BB
        action = self.gto_rules.get_preflop_action('A9o', 'button', 'open', 50)
        self.assertEqual(action, 'raise')  # Should be in range
        
        # Test 100+ BB (tighter range)
        action = self.gto_rules.get_preflop_action('A9o', 'button', 'open', 100)
        self.assertEqual(action, 'fold')  # Should fold in tighter range
    
    def test_bb_defending_ranges_all_stack_depths(self):
        """Test big blind defending ranges for all stack depths."""
        # Test 20-30 BB (wider defending)
        action = self.gto_rules.get_preflop_action('A9o', 'big_blind', 'defend', 25)
        self.assertEqual(action, 'call')  # Should defend wider
        
        # Test 50 BB
        action = self.gto_rules.get_preflop_action('A9o', 'big_blind', 'defend', 50)
        self.assertEqual(action, 'fold')  # Should fold
        
        # Test 100+ BB
        action = self.gto_rules.get_preflop_action('A9o', 'big_blind', 'defend', 100)
        self.assertEqual(action, 'fold')  # Should fold
    
    def test_3bet_ranges_all_stack_depths(self):
        """Test 3-bet ranges for all stack depths."""
        # Test button 3-bet 20-30 BB (wider)
        action = self.gto_rules.get_preflop_action('AJo', 'button', '3bet', 25)
        self.assertEqual(action, 'raise')  # Should 3-bet wider
        
        # Test button 3-bet 100+ BB
        action = self.gto_rules.get_preflop_action('AJo', 'button', '3bet', 100)
        self.assertEqual(action, 'fold')  # Should fold
    
    def test_4bet_ranges_all_stack_depths(self):
        """Test 4-bet ranges for all stack depths."""
        # Test 4-bet with premium hands
        action = self.gto_rules.get_preflop_action('AA', 'button', '4bet', 25)
        self.assertEqual(action, 'raise')
        
        action = self.gto_rules.get_preflop_action('AA', 'button', '4bet', 100)
        self.assertEqual(action, 'raise')
        
        # Test 4-bet with non-premium
        action = self.gto_rules.get_preflop_action('JJ', 'button', '4bet', 25)
        self.assertEqual(action, 'fold')
    
    def test_allin_ranges_all_stack_depths(self):
        """Test all-in ranges for all stack depths."""
        # Test 20-30 BB (wider all-in range)
        action = self.gto_rules.get_preflop_action('TT', 'button', 'allin', 25)
        self.assertEqual(action, 'raise')  # Should all-in with TT
        
        # Test 50 BB (tighter)
        action = self.gto_rules.get_preflop_action('TT', 'button', 'allin', 50)
        self.assertEqual(action, 'fold')  # Should fold
        
        # Test 100+ BB (very tight)
        action = self.gto_rules.get_preflop_action('AKs', 'button', 'allin', 100)
        self.assertEqual(action, 'fold')  # Should fold (only AA/KK all-in)
    
    def test_bet_sizing_guidelines(self):
        """Test bet sizing guidelines."""
        # Test opening sizes
        guidelines = self.gto_rules.get_bet_sizing_guidelines('open', 25)
        self.assertIn('optimal_size', guidelines)
        self.assertGreater(guidelines['optimal_size'], 0)
        
        # Test 3-bet sizes
        guidelines = self.gto_rules.get_bet_sizing_guidelines('3bet', 50)
        self.assertIn('optimal_size', guidelines)
        self.assertGreater(guidelines['optimal_size'], 0)
        
        # Test 4-bet sizes
        guidelines = self.gto_rules.get_bet_sizing_guidelines('4bet', 100)
        self.assertIn('optimal_size', guidelines)
        self.assertGreater(guidelines['optimal_size'], 0)
    
    def test_error_handling_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Invalid stack depth
        action = self.gto_rules.get_preflop_action('AA', 'button', 'open', -10)
        self.assertEqual(action, 'raise')  # Should use closest range (20 BB)
        
        action = self.gto_rules.get_preflop_action('AA', 'button', 'open', 2000)
        self.assertEqual(action, 'raise')  # Should use closest range (100 BB)
        
        # Invalid position
        action = self.gto_rules.get_preflop_action('AA', 'invalid', 'open', 100)
        self.assertEqual(action, 'fold')  # Should default to fold
        
        # Invalid action type
        action = self.gto_rules.get_preflop_action('AA', 'button', 'invalid', 100)
        self.assertEqual(action, 'fold')  # Should default to fold


if __name__ == '__main__':
    unittest.main()

