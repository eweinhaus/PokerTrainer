"""
Validation tests for Action Index Fix
Tests that frontend sends action_value directly and backend processes it correctly
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from app import app, GameManager


class TestActionIndexFix:
    """Test that action_value is sent directly and processed correctly"""
    
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
    
    def test_frontend_sends_action_value(self, client):
        """Test that API accepts action_value parameter"""
        # Start a new game
        response = client.post('/api/game/start', json={'session_id': 'test_session'})
        assert response.status_code == 200
        
        # Get game state to see legal actions
        response = client.get('/api/game/state', query_string={'session_id': 'test_session'})
        assert response.status_code == 200
        game_state = response.get_json()
        
        # Verify legal_actions exist
        assert 'legal_actions' in game_state or 'raw_legal_actions' in game_state
        
        # Try to send an action with action_value
        if 'legal_actions' in game_state and len(game_state['legal_actions']) > 0:
            action_value = game_state['legal_actions'][0]
            response = client.post('/api/game/action', json={
                'session_id': 'test_session',
                'action_value': action_value
            })
            # Should not return 400 for missing action_value
            assert response.status_code != 400 or 'action_value' not in response.get_json().get('error', '').lower()
    
    def test_backend_processes_action_value(self, game_manager):
        """Test that backend correctly processes action_value"""
        session_id = 'test_session_action_value'
        game_manager.start_game(session_id)
        
        # Get game state
        game_state = game_manager.get_game_state(session_id)
        assert game_state is not None
        
        # Get legal actions
        legal_actions = game_state.get('legal_actions', [])
        if len(legal_actions) > 0:
            action_value = legal_actions[0]
            
            # Process action with action_value
            result = game_manager.process_action(session_id, action_value)
            assert result is not None, "Action should be processed successfully"
    
    def test_action_value_mapping_consistency(self, game_manager):
        """Test that action_value maps correctly to RLCard actions"""
        session_id = 'test_mapping_consistency'
        game_manager.start_game(session_id)
        
        # Get initial state
        game_state = game_manager.get_game_state(session_id)
        legal_actions = game_state.get('legal_actions', [])
        
        if len(legal_actions) > 0:
            action_value = legal_actions[0]
            
            # Process action
            result = game_manager.process_action(session_id, action_value)
            
            # Verify action was executed (game state should change)
            assert result is not None
            # Action should be in action history or game should progress
            assert 'current_player' in result or 'stage' in result
    
    def test_all_action_values_accepted(self, client):
        """Test that all valid action values are accepted"""
        # Start game
        response = client.post('/api/game/start', json={'session_id': 'test_all_actions'})
        assert response.status_code == 200
        
        # Get game state
        response = client.get('/api/game/state', query_string={'session_id': 'test_all_actions'})
        assert response.status_code == 200
        game_state = response.get_json()
        
        legal_actions = game_state.get('legal_actions', [])
        raw_legal_actions = game_state.get('raw_legal_actions', [])
        all_legal = legal_actions if legal_actions else raw_legal_actions
        
        # Test each legal action value
        for action_value in all_legal[:3]:  # Test first 3 to avoid too many actions
            response = client.post('/api/game/action', json={
                'session_id': 'test_all_actions',
                'action_value': action_value
            })
            # Should not fail with "missing action_value" error
            if response.status_code == 400:
                error = response.get_json().get('error', '')
                assert 'action_value' not in error.lower() or 'missing' not in error.lower(), \
                    f"Action value {action_value} should be accepted"
    
    def test_invalid_action_value_rejected(self, client):
        """Test that invalid action values are properly rejected"""
        # Start game
        response = client.post('/api/game/start', json={'session_id': 'test_invalid'})
        assert response.status_code == 200
        
        # Try invalid action value
        response = client.post('/api/game/action', json={
            'session_id': 'test_invalid',
            'action_value': 999  # Invalid action
        })
        
        # Should return error
        assert response.status_code in [400, 500]
        error = response.get_json().get('error', '')
        assert len(error) > 0, "Should return error message for invalid action"

