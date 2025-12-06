"""
Unit tests for LLMOpponentAgent context building components.

Tests opponent card extraction, action history reconstruction, and context building.
"""

import pytest
import sys
import os

# Add parent directory to path
# Add webapp directory to path for coach imports
sys.path.append('webapp')

from coach.llm_opponent_agent import LLMOpponentAgent, Action


class TestOpponentCardExtraction:
    """Tests for _extract_opponent_cards()"""
    
    def test_extract_valid_cards(self):
        """Test extraction of valid opponent cards"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13]  # Two aces
            }
        }
        
        cards = agent._extract_opponent_cards(state)
        assert cards == [0, 13]
        assert len(cards) == 2
    
    def test_extract_missing_hand(self):
        """Test handling of missing hand"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {}
        }
        
        cards = agent._extract_opponent_cards(state)
        assert cards is None
    
    def test_extract_invalid_hand(self):
        """Test handling of invalid hand (wrong length)"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0]  # Only one card
            }
        }
        
        cards = agent._extract_opponent_cards(state)
        assert cards is None
    
    def test_extract_empty_hand(self):
        """Test handling of empty hand"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': []
            }
        }
        
        cards = agent._extract_opponent_cards(state)
        assert cards is None


class TestActionHistoryReconstruction:
    """Tests for _build_action_history()"""
    
    def test_build_action_history_preflop(self):
        """Test action history building for preflop"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 9,  # SB (1) + BB (2) + raise (6) = 9
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],
                'stage': 0  # Preflop
            }
        }
        
        history = agent._build_action_history(state)
        
        # Should have at least blinds
        assert len(history) >= 2
        assert history[0]['action'] == 'blind'
        assert history[1]['action'] == 'blind'
    
    def test_build_action_history_postflop(self):
        """Test action history building for postflop"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [26, 27, 28],  # Flop
                'pot': 100,
                'big_blind': 2,
                'all_chips': [950, 1050],
                'raised': [0, 0],
                'stage': 1  # Flop
            }
        }
        
        history = agent._build_action_history(state)
        
        # Should have at least blinds
        assert len(history) >= 2
        assert history[0]['stage'] == 'preflop'
    
    def test_build_action_history_all_in(self):
        """Test action history building with all-in scenario"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 2000,  # Large pot (all-in)
                'big_blind': 2,
                'all_chips': [0, 0],  # Both players all-in
                'raised': [1000, 1000],
                'stage': 0
            }
        }
        
        history = agent._build_action_history(state)
        
        # Should have action history
        assert len(history) >= 2


class TestContextBuilding:
    """Tests for _build_context()"""
    
    def test_build_context_preflop(self):
        """Test context building for preflop"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],  # Opponent cards
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT]
        }
        
        context = agent._build_context(state)
        
        # Verify required fields
        assert 'opponent_cards' in context
        assert 'user_position' in context
        assert 'opponent_position' in context
        assert 'current_stage' in context
        assert 'public_cards' in context
        assert 'pot_size' in context
        assert 'pot_size_bb' in context
        assert 'big_blind' in context
        assert 'current_stacks' in context
        assert 'stack_depths' in context
        assert 'action_history' in context
        assert 'legal_actions' in context
        assert 'legal_actions_labels' in context
        assert 'facing_bet' in context
        assert 'bet_to_call' in context
        assert 'bet_to_call_bb' in context
        assert 'pot_odds' in context
        
        # Verify values
        assert context['current_stage'] == 'preflop'
        assert len(context['opponent_cards']) == 2
        assert len(context['public_cards']) == 0
        assert context['user_position'] == 'button'
        assert context['opponent_position'] == 'big_blind'
        assert context['pot_size'] == 9
        assert context['big_blind'] == 2
    
    def test_build_context_postflop(self):
        """Test context building for postflop"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [26, 27, 28],  # Flop
                'pot': 100,
                'big_blind': 2,
                'all_chips': [950, 1050],
                'raised': [0, 0],
                'stage': 1  # Flop
            },
            'raw_legal_actions': [Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT]
        }
        
        context = agent._build_context(state)
        
        assert context['current_stage'] == 'flop'
        assert len(context['public_cards']) == 3
        assert context['facing_bet'] == False  # No bet to call
    
    def test_build_context_facing_bet(self):
        """Test context building when facing a bet"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],  # User raised to 6, opponent at 2
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT]
        }
        
        context = agent._build_context(state)
        
        # Opponent is facing a bet (user raised to 6, opponent at 2)
        # Actually wait - in this state, opponent_raised=2, our_raised=6
        # So opponent_raised (2) < our_raised (6), so facing_bet should be True
        # But wait, we're the opponent (player 1), so our_raised is raised[1] = 2
        # and opponent_raised is raised[0] = 6
        # So facing_bet = opponent_raised > our_raised = 6 > 2 = True
        assert context['facing_bet'] == True
        assert context['bet_to_call'] > 0
        assert context['pot_odds'] > 0
    
    def test_build_context_all_in(self):
        """Test context building with all-in scenario"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 2000,
                'big_blind': 2,
                'all_chips': [0, 0],
                'raised': [1000, 1000],
                'stage': 0
            },
            'raw_legal_actions': [Action.CHECK_CALL]  # Only call available
        }
        
        context = agent._build_context(state)
        
        assert context['pot_size'] == 2000
        assert Action.ALL_IN not in context['legal_actions']  # Not available
    
    def test_build_context_legal_actions_labels(self):
        """Test that legal actions labels are correctly built"""
        agent = LLMOpponentAgent(num_actions=5)
        
        state = {
            'raw_obs': {
                'hand': [0, 13],
                'public_cards': [],
                'pot': 9,
                'big_blind': 2,
                'all_chips': [994, 1000],
                'raised': [6, 2],
                'stage': 0
            },
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
        }
        
        context = agent._build_context(state)
        
        labels = context['legal_actions_labels']
        # Updated to match new ActionLabeling module output
        assert labels[Action.FOLD] == 'Fold'
        assert labels[Action.CHECK_CALL] in ['Check', 'Call']  # Context-aware
        assert 'Bet' in labels[Action.RAISE_HALF_POT] or 'Raise' in labels[Action.RAISE_HALF_POT]
        assert 'Raise' in labels[Action.RAISE_POT] or 'Bet' in labels[Action.RAISE_POT]
        assert labels[Action.ALL_IN] == 'All-In'


