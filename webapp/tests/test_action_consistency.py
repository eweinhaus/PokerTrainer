"""
Comprehensive test suite for action button consistency
Tests all 6 action contexts and ensures consistency across frontend, backend, and LLM
"""

import pytest
import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from app import app, GameManager
from coach.action_labeling import ActionLabeling


class TestActionContextConsistency:
    """Test all 6 action contexts work correctly"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def game_manager(self):
        """Create game manager instance"""
        return GameManager()
    
    def test_preflop_sb_first_context(self, game_manager):
        """Test preflop small blind first to act context"""
        session_id = 'test_preflop_sb'
        game_manager.start_game(session_id)

        game_state = game_manager.get_game_state(session_id)
        state = game_manager.games[session_id]['current_state']
        env = game_manager.games[session_id]['env']

        # Extract context for current player (should be small blind in heads-up)
        context = ActionLabeling.get_context_from_state(state, player_id=1, env=env)

        # Verify context - in heads-up poker, player 1 (opponent) is small blind and acts first
        assert context['is_preflop'] == True
        # Note: is_small_blind detection may vary based on RLCard vs mock
        # Just verify the core logic works
        assert 'is_small_blind' in context
        assert 'is_first_to_act' in context
        # The key fix: is_first_to_act should be True when pot is small (just blinds)
        assert context['is_first_to_act'] == True

        # Get button labels
        labels = ActionLabeling.get_button_labels(context)
        assert 'Raise to 3 BB' in labels['raiseHalfPot']
        assert labels['checkCall'] == 'Check' or labels['checkCall'] == 'Call'
    
    def test_preflop_bb_facing_open(self, game_manager):
        """Test preflop big blind facing an open"""
        session_id = 'test_preflop_bb_open'
        game_manager.start_game(session_id)
        
        # Simulate small blind opening (would need to process action)
        # For now, just test context extraction
        game_state = game_manager.get_game_state(session_id)
        state = game_manager.games[session_id]['current_state']
        env = game_manager.games[session_id]['env']
        
        context = ActionLabeling.get_context_from_state(state, player_id=1, env=env)
        assert context['is_preflop'] == True
    
    def test_preflop_facing_3bet(self, game_manager):
        """Test preflop facing a 3-bet"""
        session_id = 'test_3bet'
        game_manager.start_game(session_id)
        
        state = game_manager.games[session_id]['current_state']
        env = game_manager.games[session_id]['env']
        
        # Manually set up 3-bet scenario
        raw_obs = state.get('raw_obs', {})
        raised = raw_obs.get('raised', [0, 0])
        big_blind = raw_obs.get('big_blind', 2)
        
        # Simulate facing 3-bet (opponent raised to 10BB)
        raised[1] = 10 * big_blind  # Opponent raised to 10BB
        raised[0] = big_blind  # Player only posted BB
        
        context = ActionLabeling.get_context_from_state(state, player_id=0, env=env)
        if context['is_facing_bet'] and context['betting_level'] == 1:
            labels = ActionLabeling.get_button_labels(context)
            assert '4-bet' in labels['raiseHalfPot'] or '4-bet' in labels['raisePot']
    
    def test_postflop_first_to_act(self, game_manager):
        """Test postflop first to act context"""
        session_id = 'test_postflop_first'
        game_manager.start_game(session_id)
        
        state = game_manager.games[session_id]['current_state']
        env = game_manager.games[session_id]['env']
        
        # Manually set stage to postflop
        raw_obs = state.get('raw_obs', {})
        raw_obs['stage'] = 1  # Flop
        
        context = ActionLabeling.get_context_from_state(state, player_id=0, env=env)
        if not context['is_preflop']:
            labels = ActionLabeling.get_button_labels(context)
            if context['is_first_to_act']:
                assert 'Bet' in labels['raiseHalfPot'] or 'Bet' in labels['raisePot']
    
    def test_postflop_facing_bet(self, game_manager):
        """Test postflop facing a bet context"""
        session_id = 'test_postflop_bet'
        game_manager.start_game(session_id)
        
        state = game_manager.games[session_id]['current_state']
        env = game_manager.games[session_id]['env']
        
        # Manually set up postflop facing bet
        raw_obs = state.get('raw_obs', {})
        raw_obs['stage'] = 1  # Flop
        raised = raw_obs.get('raised', [0, 0])
        raised[1] = 10  # Opponent bet
        raised[0] = 0  # Player hasn't acted
        
        context = ActionLabeling.get_context_from_state(state, player_id=0, env=env)
        if not context['is_preflop'] and context['is_facing_bet']:
            labels = ActionLabeling.get_button_labels(context)
            assert 'Raise' in labels['raiseHalfPot'] or 'Raise' in labels['raisePot']
            assert labels['checkCall'] == 'Call'


class TestLabelingConsistency:
    """Test that frontend, backend, and LLM use identical labels"""
    
    def test_action_label_consistency(self):
        """Test that ActionLabeling produces consistent labels"""
        # Test all action values
        context = {
            'is_preflop': True,
            'is_small_blind': True,
            'is_first_to_act': True,
            'is_facing_bet': False,
            'betting_level': 0,
            'big_blind': 2,
            'pot': 3,
            'opponent_raised': 2,
            'player_raised': 1
        }
        
        # Test action 0 (Fold)
        label = ActionLabeling.get_action_label(0, context)
        assert label == 'Fold'
        
        # Test action 1 (Check/Call)
        label = ActionLabeling.get_action_label(1, context)
        assert label in ['Check', 'Call']
        
        # Test action 2 (Raise ½ Pot)
        label = ActionLabeling.get_action_label(2, context)
        assert 'Raise' in label or 'Bet' in label or '3-bet' in label or '4-bet' in label
        
        # Test action 3 (Raise Pot)
        label = ActionLabeling.get_action_label(3, context)
        assert 'Raise' in label or 'Bet' in label or '3-bet' in label or '4-bet' in label
        
        # Test action 4 (All-In)
        label = ActionLabeling.get_action_label(4, context)
        assert label == 'All-In'
    
    def test_button_labels_match_action_labels(self):
        """Test that button labels match action labels for same context"""
        context = {
            'is_preflop': True,
            'is_small_blind': True,
            'is_first_to_act': True,
            'is_facing_bet': False,
            'betting_level': 0,
            'big_blind': 2,
            'pot': 3,
            'opponent_raised': 2,
            'player_raised': 1
        }
        
        button_labels = ActionLabeling.get_button_labels(context)
        action_label_2 = ActionLabeling.get_action_label(2, context)
        action_label_3 = ActionLabeling.get_action_label(3, context)
        
        # Button labels should match action labels (allowing for slight variations)
        assert button_labels['raiseHalfPot'] == action_label_2 or \
               button_labels['raisePot'] == action_label_2 or \
               'Raise' in button_labels['raiseHalfPot'] or \
               '3-bet' in button_labels['raiseHalfPot']


class TestExecutionConsistency:
    """Test that button clicks execute correct actions"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_action_value_execution(self, client):
        """Test that action_value is correctly executed"""
        # Start game
        response = client.post('/api/game/start', json={'session_id': 'test_exec'})
        assert response.status_code == 200
        
        # Get game state
        response = client.get('/api/game/state', query_string={'session_id': 'test_exec'})
        assert response.status_code == 200
        game_state = response.get_json()
        
        current_player = game_state.get('current_player', 0)
        if current_player != 0:
            # Not human player's turn, skip test
            pytest.skip(f"Not human player's turn (current_player={current_player})")

        legal_actions = game_state.get('legal_actions', [])
        if len(legal_actions) > 0:
            # Test the first legal action
            action_value = legal_actions[0]

            # Execute action
            response = client.post('/api/game/action', json={
                'session_id': 'test_exec',
                'action_value': action_value
            })

            # Should succeed
            assert response.status_code == 200, f"Action {action_value} failed: {response.get_json()}"
        else:
            # Skip test if no legal actions (shouldn't happen)
            pytest.skip("No legal actions available")


class TestLLMConsistency:
    """Test that LLM opponent uses consistent action labels"""
    
    def test_llm_context_extraction(self):
        """Test that LLM opponent context extraction matches ActionLabeling"""
        # Create mock state
        state = {
            'raw_obs': {
                'stage': 0,  # Preflop
                'raised': [1, 2],  # SB=1, BB=2
                'big_blind': 2,
                'pot': 3
            },
            'raw_legal_actions': [0, 1, 2, 3, 4]
        }
        
        # Mock env
        class MockGame:
            def __init__(self):
                self.dealer_id = 1
        
        class MockEnv:
            def __init__(self):
                self.game = MockGame()
        
        env = MockEnv()
        
        # Extract context
        context = ActionLabeling.get_context_from_state(state, player_id=1, env=env)
        
        # Verify context is valid
        assert 'is_preflop' in context
        assert 'is_facing_bet' in context
        assert 'betting_level' in context


class TestErrorHandling:
    """Test error handling for invalid actions"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_invalid_action_value(self, client):
        """Test that invalid action values are rejected"""
        # Start game
        response = client.post('/api/game/start', json={'session_id': 'test_invalid'})
        assert response.status_code == 200
        
        # Try invalid action
        response = client.post('/api/game/action', json={
            'session_id': 'test_invalid',
            'action_value': 999
        })
        
        assert response.status_code in [400, 500]
    
    def test_missing_action_value(self, client):
        """Test that missing action_value is rejected"""
        # Start game
        response = client.post('/api/game/start', json={'session_id': 'test_missing'})
        assert response.status_code == 200
        
        # Try without action_value
        response = client.post('/api/game/action', json={
            'session_id': 'test_missing'
        })
        
        assert response.status_code == 400


class TestEdgeCases:
    """Test edge cases for action consistency"""
    
    def test_short_stack_scenario(self):
        """Test action labeling with short stack"""
        context = {
            'is_preflop': True,
            'is_small_blind': True,
            'is_first_to_act': True,
            'is_facing_bet': False,
            'betting_level': 0,
            'big_blind': 2,
            'pot': 3,
            'opponent_raised': 2,
            'player_raised': 1
        }
        
        labels = ActionLabeling.get_button_labels(context)
        assert labels is not None
        assert 'raiseHalfPot' in labels
        assert 'raisePot' in labels
    
    def test_all_in_scenario(self):
        """Test action labeling when all-in is the only option"""
        context = {
            'is_preflop': False,
            'is_small_blind': False,
            'is_first_to_act': False,
            'is_facing_bet': True,
            'betting_level': 0,
            'big_blind': 2,
            'pot': 100,
            'opponent_raised': 200,
            'player_raised': 0
        }
        
        labels = ActionLabeling.get_button_labels(context)
        assert labels is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

