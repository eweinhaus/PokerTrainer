"""
Unit tests for StrategyEvaluator Phase 2 enhancements
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from coach.strategy_evaluator import StrategyEvaluator
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from webapp.coach.strategy_evaluator import StrategyEvaluator


class TestStrategyEvaluatorPhase2(unittest.TestCase):
    """Test StrategyEvaluator Phase 2 enhancements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = StrategyEvaluator()
    
    def test_continuation_betting_evaluation(self):
        """Test continuation betting evaluation."""
        hand_strength = {'category': 'top_pair_plus', 'description': 'Top pair'}
        board_texture = {'type': 'dry', 'draws': [], 'coordination': 'low'}
        game_state = {'pot': 100, 'big_blind': 2}
        
        eval_result = self.evaluator.evaluate_continuation_bet(
            hand_strength, board_texture, game_state
        )
        
        self.assertIn('should_cbet', eval_result)
        self.assertIn('frequency', eval_result)
        self.assertIn('optimal_size', eval_result)
        self.assertIn('explanation', eval_result)
    
    def test_value_vs_bluff_evaluation(self):
        """Test value vs bluff evaluation."""
        hand_strength = {'category': 'top_pair_plus', 'description': 'Top pair'}
        board_texture = {'type': 'dry', 'draws': [], 'coordination': 'low'}
        opponent_range = {'AA': 1.0, 'KK': 1.0}
        game_state = {'pot': 100}
        
        eval_result = self.evaluator.distinguish_value_vs_bluff(
            hand_strength, board_texture, opponent_range, game_state
        )
        
        self.assertIn('type', eval_result)
        self.assertIn(eval_result['type'], ['value', 'bluff', 'check'])
        self.assertIn('explanation', eval_result)
    
    def test_preflop_bet_sizing_evaluation(self):
        """Test preflop bet sizing evaluation."""
        action_type = 'open'
        action = {'bet_amount': 5}  # 2.5 BB if BB=2
        game_state = {'big_blind': 2, 'pot': 0}
        stack_depth = 100
        
        eval_result = self.evaluator.evaluate_preflop_bet_sizing(
            action_type, action, game_state, stack_depth
        )
        
        self.assertIsNotNone(eval_result)
        if eval_result:
            self.assertIn('grade', eval_result)
            self.assertIn('grade_percentage', eval_result)
            self.assertIn('explanation', eval_result)
    
    def test_postflop_bet_sizing_evaluation(self):
        """Test postflop bet sizing evaluation."""
        action_type = 'cbet'
        action = {'bet_amount': 50}  # 0.5x pot if pot=100
        game_state = {'pot': 100, 'big_blind': 2}
        board_texture = {'type': 'dry', 'draws': [], 'coordination': 'low'}
        hand_strength = {'category': 'top_pair_plus', 'description': 'Top pair'}
        
        eval_result = self.evaluator.evaluate_postflop_bet_sizing(
            action_type, action, game_state, board_texture, hand_strength
        )
        
        self.assertIsNotNone(eval_result)
        if eval_result:
            self.assertIn('grade', eval_result)
            self.assertIn('grade_percentage', eval_result)
            self.assertIn('explanation', eval_result)
    
    def test_grading_with_position_adjustments(self):
        """Test grading with position adjustments."""
        # Test button position (more lenient)
        grade_button = self.evaluator.calculate_grade(85, position='button')
        grade_bb = self.evaluator.calculate_grade(85, position='big_blind')
        
        # Button should have slightly higher percentage
        self.assertGreaterEqual(grade_button['percentage'], grade_bb['percentage'] - 3)
    
    def test_grading_with_stack_depth_adjustments(self):
        """Test grading with stack depth adjustments."""
        # Test short stack (more lenient)
        grade_short = self.evaluator.calculate_grade(85, stack_depth=25)
        grade_deep = self.evaluator.calculate_grade(85, stack_depth=150)
        
        # Short stack should have slightly higher percentage
        self.assertGreaterEqual(grade_short['percentage'], grade_deep['percentage'] - 5)
    
    def test_enhanced_explanation_generation(self):
        """Test enhanced explanation generation."""
        evaluation_data = {
            'stage': 'preflop',
            'hand': 'AA',
            'position': 'button',
            'action_type': 'open',
            'player_action': 'Raise',
            'optimal_action': 'Raise',
            'grade': 'A',
            'optimality_score': 95,
            'stack_depth': 100
        }
        
        explanation = self.evaluator.generate_explanation(evaluation_data)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 50)  # Should be detailed
        self.assertIn('button', explanation.lower())  # Should mention position


if __name__ == '__main__':
    unittest.main()

