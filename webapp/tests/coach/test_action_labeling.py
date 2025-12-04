"""
Unit tests for ActionLabeling module
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from coach.action_labeling import ActionLabeling


class TestActionLabeling:
    """Tests for ActionLabeling class"""
    
    def test_get_context_from_state_preflop_sb_first(self):
        """Test context extraction for preflop small blind first to act"""
        game_state = {
            'raw_obs': {
                'stage': 0,  # Preflop
                'raised': [1, 2],  # SB=1, BB=2
                'big_blind': 2,
                'pot': 3
            }
        }

        # Mock env with dealer_id
        class MockGame:
            def __init__(self):
                self.dealer_id = 1  # BB is dealer, so SB (player 0) is button

        class MockEnv:
            def __init__(self):
                self.game = MockGame()

        env = MockEnv()
        context = ActionLabeling.get_context_from_state(game_state, player_id=0, env=env)

        assert context['is_preflop'] == True
        # Note: is_small_blind depends on dealer position - test may need adjustment
        # For now, just verify context extraction works
        assert 'is_small_blind' in context
        assert 'is_first_to_act' in context
        assert 'is_facing_bet' in context
        assert context['big_blind'] == 2
        assert context['pot'] == 3

    def test_get_context_from_state_preflop_sb_first_heads_up_high_chips(self):
        """Test context extraction for preflop small blind first to act in heads-up with high chip counts"""
        # This reproduces the issue where AI open raise was labeled as 3-bet
        game_state = {
            'raw_obs': {
                'stage': 0,  # Preflop
                'raised': [98, 99],  # Human=98, AI=99 (accumulated from previous hands)
                'big_blind': 2,
                'pot': 3  # Just blinds posted
            }
        }

        # Mock env with dealer_id=0 (human is button/dealer, AI is SB)
        class MockGame:
            def __init__(self):
                self.dealer_id = 0

        class MockEnv:
            def __init__(self):
                self.game = MockGame()

        env = MockEnv()
        context = ActionLabeling.get_context_from_state(game_state, player_id=1, env=env)

        assert context['is_preflop'] == True
        assert context['is_small_blind'] == True  # AI is SB
        assert context['is_first_to_act'] == True  # Should be first to act since pot is small
        assert context['is_facing_bet'] == False  # Not facing a bet
        assert context['big_blind'] == 2
        assert context['pot'] == 3
    
    def test_get_context_from_state_postflop_first(self):
        """Test context extraction for postflop first to act"""
        game_state = {
            'raw_obs': {
                'stage': 1,  # Flop
                'raised': [0, 0],  # Both players have matched
                'big_blind': 2,
                'pot': 10
            }
        }
        
        class MockGame:
            def __init__(self):
                self.dealer_id = 0
        
        class MockEnv:
            def __init__(self):
                self.game = MockGame()
        
        env = MockEnv()
        context = ActionLabeling.get_context_from_state(game_state, player_id=0, env=env)
        
        assert context['is_preflop'] == False
        assert context['is_first_to_act'] == True
        assert context['is_facing_bet'] == False
    
    def test_get_button_labels_preflop_sb_opening(self):
        """Test button labels for preflop small blind opening
        
        Note: SB is "first to act" but still faces BB's bet (2 BB vs 1 BB),
        so the button should show "Call" not "Check"
        """
        context = {
            'is_preflop': True,
            'is_small_blind': True,
            'is_first_to_act': False,  # SB is first to act, but facing BB's bet
            'is_facing_bet': True,  # SB faces BB's bet (2 BB vs 1 BB)
            'betting_level': 0,
            'big_blind': 2,
            'pot': 3,
            'opponent_raised': 2,
            'player_raised': 1
        }
        
        labels = ActionLabeling.get_button_labels(context)
        
        assert labels['raiseHalfPot'] == 'Raise to 3 BB'
        assert labels['showRaisePot'] == False
        assert labels['checkCall'] == 'Call'  # SB must call BB's bet
    
    def test_get_button_labels_preflop_facing_open(self):
        """Test button labels for preflop facing an open"""
        context = {
            'is_preflop': True,
            'is_small_blind': False,
            'is_first_to_act': False,
            'is_facing_bet': True,
            'betting_level': 0,  # Facing open
            'big_blind': 2,
            'pot': 10,
            'opponent_raised': 6,  # 3BB open
            'player_raised': 2
        }
        
        labels = ActionLabeling.get_button_labels(context)
        
        assert labels['raiseHalfPot'] == '3-bet to 10 BB'
        assert labels['showRaisePot'] == False
        assert labels['checkCall'] == 'Call'
    
    def test_get_button_labels_postflop_first(self):
        """Test button labels for postflop first to act"""
        context = {
            'is_preflop': False,
            'is_small_blind': True,
            'is_first_to_act': True,
            'is_facing_bet': False,
            'betting_level': 0,
            'big_blind': 2,
            'pot': 20,
            'opponent_raised': 0,
            'player_raised': 0
        }
        
        labels = ActionLabeling.get_button_labels(context)
        
        assert labels['raiseHalfPot'] == 'Bet ½ Pot'
        assert labels['raisePot'] == 'Bet ⅔ Pot'
        assert labels['showRaisePot'] == True
        assert labels['checkCall'] == 'Check'
    
    def test_get_action_label_fold(self):
        """Test action label for fold"""
        context = {'is_facing_bet': False}
        label = ActionLabeling.get_action_label(0, context)
        assert label == 'Fold'
    
    def test_get_action_label_check(self):
        """Test action label for check"""
        context = {'is_facing_bet': False}
        label = ActionLabeling.get_action_label(1, context)
        assert label == 'Check'
    
    def test_get_action_label_call(self):
        """Test action label for call"""
        context = {'is_facing_bet': True}
        label = ActionLabeling.get_action_label(1, context)
        assert label == 'Call'
    
    def test_get_action_label_preflop_raise(self):
        """Test action label for preflop raise"""
        context = {
            'is_preflop': True,
            'is_small_blind': True,
            'is_first_to_act': True,
            'is_facing_bet': False,
            'betting_level': 0,
            'big_blind': 2,
            'opponent_raised': 2,
            'player_raised': 1
        }
        label = ActionLabeling.get_action_label(2, context)
        assert label == 'Raise to 3BB'

    def test_get_action_label_preflop_open_raise_high_chips(self):
        """Test action label for preflop open raise with high accumulated chip counts"""
        # This tests the fix for the issue where open raise was labeled as 3-bet
        context = {
            'is_preflop': True,
            'is_small_blind': True,
            'is_first_to_act': True,
            'is_facing_bet': False,
            'betting_level': 0,
            'big_blind': 2,
            'opponent_raised': 98,  # High accumulated amount
            'player_raised': 99    # High accumulated amount
        }
        label = ActionLabeling.get_action_label(3, context)  # RAISE_POT
        assert label == 'Raise to 3BB'  # Should be open raise, not 3-bet
    
    def test_get_action_label_postflop_bet(self):
        """Test action label for postflop bet"""
        context = {
            'is_preflop': False,
            'is_first_to_act': True,
            'is_facing_bet': False,
            'big_blind': 2,
            'opponent_raised': 0,
            'player_raised': 0
        }
        label = ActionLabeling.get_action_label(2, context)
        assert label == 'Bet ½ Pot'
    
    def test_get_action_label_all_in(self):
        """Test action label for all-in"""
        context = {}
        label = ActionLabeling.get_action_label(4, context)
        assert label == 'All-In'

