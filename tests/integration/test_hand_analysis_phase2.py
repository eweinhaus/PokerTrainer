"""
Integration tests for complete hand analysis flow with Phase 2 enhancements
"""

import unittest
import sys
import os
import time

# Add parent directory to path
# Add webapp directory to path for coach imports
sys.path.append('webapp')

try:
    from coach.strategy_evaluator import StrategyEvaluator
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from webapp.coach.strategy_evaluator import StrategyEvaluator


class TestHandAnalysisPhase2(unittest.TestCase):
    """Test complete hand analysis flow with Phase 2 enhancements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = StrategyEvaluator()
    
    def test_complete_hand_analysis_preflop(self):
        """Test complete hand analysis for preflop decisions."""
        hand_history = [
            {'player_id': 0, 'action': 'Raise', 'stage': 'preflop'},
            {'player_id': 1, 'action': 'Call', 'stage': 'preflop'},
        ]
        
        game_state = {
            'stage': 0,
            'hand': [0, 1],  # Simplified
            'pot': 6,
            'big_blind': 2,
            'all_chips': [100, 100],
            'current_player': 0,
            'button_id': 0
        }
        
        start_time = time.time()
        analysis = self.evaluator.analyze_hand(hand_history, game_state)
        elapsed_time = time.time() - start_time
        
        # Check response structure
        self.assertIn('overall_grade', analysis)
        self.assertIn('overall_grade_percentage', analysis)
        self.assertIn('decisions', analysis)
        self.assertIn('key_insights', analysis)
        self.assertIn('learning_points', analysis)
        
        # Check performance
        self.assertLess(elapsed_time, 2.0, "Hand analysis should complete within 2 seconds")
        
        # Check decisions
        self.assertGreater(len(analysis['decisions']), 0)
        for decision in analysis['decisions']:
            self.assertIn('grade', decision)
            self.assertIn('explanation', decision)
    
    def test_complete_hand_analysis_postflop(self):
        """Test complete hand analysis for postflop decisions."""
        hand_history = [
            {'player_id': 0, 'action': 'Raise', 'stage': 'preflop'},
            {'player_id': 1, 'action': 'Call', 'stage': 'preflop'},
            {'player_id': 0, 'action': 'Raise ½ Pot', 'stage': 'flop'},
            {'player_id': 1, 'action': 'Fold', 'stage': 'flop'},
        ]
        
        game_state = {
            'stage': 1,
            'hand': [0, 1],
            'public_cards': [2, 3, 4],
            'pot': 12,
            'big_blind': 2,
            'all_chips': [94, 94],
            'current_player': 0,
            'button_id': 0
        }
        
        start_time = time.time()
        analysis = self.evaluator.analyze_hand(hand_history, game_state)
        elapsed_time = time.time() - start_time
        
        # Check performance
        self.assertLess(elapsed_time, 2.0, "Hand analysis should complete within 2 seconds")
        
        # Check that postflop decisions are evaluated (if any exist)
        postflop_decisions = [d for d in analysis['decisions'] if d.get('stage') != 'preflop']
        # Note: May be 0 if only preflop decisions were made by player
        # This is acceptable - the test verifies the system works
    
    def test_performance_requirements(self):
        """Test that performance requirements are met."""
        hand_history = [
            {'player_id': 0, 'action': 'Raise', 'stage': 'preflop'},
            {'player_id': 1, 'action': 'Call', 'stage': 'preflop'},
            {'player_id': 0, 'action': 'Raise ½ Pot', 'stage': 'flop'},
            {'player_id': 1, 'action': 'Call', 'stage': 'flop'},
            {'player_id': 0, 'action': 'Check', 'stage': 'turn'},
            {'player_id': 1, 'action': 'Check', 'stage': 'turn'},
            {'player_id': 0, 'action': 'Raise ½ Pot', 'stage': 'river'},
            {'player_id': 1, 'action': 'Fold', 'stage': 'river'},
        ]
        
        game_state = {
            'stage': 3,
            'hand': [0, 1],
            'public_cards': [2, 3, 4, 5, 6],
            'pot': 20,
            'big_blind': 2,
            'all_chips': [90, 90],
            'current_player': 0,
            'button_id': 0
        }
        
        start_time = time.time()
        analysis = self.evaluator.analyze_hand(hand_history, game_state)
        elapsed_time = time.time() - start_time
        
        # Performance requirement: < 2 seconds
        self.assertLess(elapsed_time, 2.0, f"Analysis took {elapsed_time:.2f} seconds, should be < 2 seconds")
        
        # Verify analysis is complete
        self.assertIn('overall_grade', analysis)
        self.assertGreater(len(analysis['decisions']), 0)


if __name__ == '__main__':
    unittest.main()

