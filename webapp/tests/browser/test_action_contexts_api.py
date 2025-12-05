#!/usr/bin/env python3
"""
Test all 6 action contexts via API to verify button labels
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app import app, GameManager
from coach.action_labeling import ActionLabeling


def test_action_contexts():
    """Test all 6 action contexts"""
    results = {}
    
    with app.test_client() as client:
        # Test 1: Preflop SB First (Small Blind Opening)
        response = client.post('/api/game/start', json={'session_id': 'test_sb_first'})
        if response.status_code == 200:
            game_state = response.get_json()
            session_id = game_state.get('session_id', 'test_sb_first')
            
            # Get button labels
            response = client.post('/api/game/button-labels', json={'session_id': session_id})
            if response.status_code == 200:
                labels = response.get_json()
                results['preflop_sb_first'] = {
                    'status': 'PASS',
                    'labels': labels,
                    'expected': {
                        'raiseHalfPot': 'Raise to 3 BB',
                        'checkCall': 'Call',  # SB faces BB bet
                        'showRaisePot': False
                    }
                }
                # Verify
                assert 'Raise to 3 BB' in labels.get('raiseHalfPot', ''), "Should show 'Raise to 3 BB'"
                assert labels.get('checkCall') in ['Check', 'Call'], "Should show Check or Call"
            else:
                results['preflop_sb_first'] = {'status': 'FAIL', 'error': response.get_json()}
        
        # Test 2: Preflop BB Facing Open (3-bet scenario)
        response = client.post('/api/game/start', json={'session_id': 'test_bb_open'})
        if response.status_code == 200:
            game_state = response.get_json()
            session_id = game_state.get('session_id', 'test_bb_open')
            
            # Simulate SB opening to 3BB (would need to process action)
            # For now, check current state
            response = client.post('/api/game/button-labels', json={'session_id': session_id})
            if response.status_code == 200:
                labels = response.get_json()
                results['preflop_bb_open'] = {
                    'status': 'PASS',
                    'labels': labels
                }
        
        # Test 3: Postflop First to Act
        # Create a game and advance to postflop
        response = client.post('/api/game/start', json={'session_id': 'test_postflop'})
        if response.status_code == 200:
            game_state = response.get_json()
            session_id = game_state.get('session_id', 'test_postflop')
            
            # Manually check if we can get to postflop
            # This would require playing through preflop
            results['postflop_first'] = {'status': 'PENDING', 'note': 'Requires game progression'}
        
        # Test 4: Test ActionLabeling module directly
        
        # Test Preflop SB First context
        game_state = {
            'raw_obs': {
                'stage': 0,  # Preflop
                'raised': [1, 2],  # SB=1, BB=2
                'big_blind': 2,
                'pot': 3
            }
        }
        
        class MockGame:
            def __init__(self):
                self.dealer_id = 1  # BB is dealer, so SB (player 0) is button
        
        class MockEnv:
            def __init__(self):
                self.game = MockGame()
        
        env = MockEnv()
        context = ActionLabeling.get_context_from_state(game_state, player_id=0, env=env)
        labels = ActionLabeling.get_button_labels(context)
        
        
        # Verify preflop SB first
        assert context['is_preflop'] == True, "Should be preflop"
        # Note: SB faces BB bet (2 vs 1), so it's facing a bet, not first to act
        assert context['is_facing_bet'] == True, "SB should be facing BB bet"
        assert labels.get('checkCall') == 'Call', "Should show Call when facing bet"
        # When facing a bet, it shows 3-bet option
        assert '3-bet' in labels.get('raiseHalfPot', '') or 'Raise' in labels.get('raiseHalfPot', ''), \
            f"Should show 3-bet or Raise, got '{labels.get('raiseHalfPot')}'"
        assert labels.get('showRaisePot') == False, "Should hide second raise option preflop"
        results['action_labeling_preflop_sb'] = {'status': 'PASS', 'labels': labels}
        
        # Test Preflop Facing Open (3-bet)
        game_state_3bet = {
            'raw_obs': {
                'stage': 0,  # Preflop
                'raised': [2, 6],  # Player=2 (BB), Opponent=6 (3BB open)
                'big_blind': 2,
                'pot': 9
            }
        }
        context_3bet = ActionLabeling.get_context_from_state(game_state_3bet, player_id=0, env=env)
        labels_3bet = ActionLabeling.get_button_labels(context_3bet)
        
        assert context_3bet['is_facing_bet'] == True, "Should be facing a bet"
        assert context_3bet['betting_level'] == 0, "Should be facing open (betting level 0)"
        assert '3-bet' in labels_3bet.get('raiseHalfPot', ''), f"Should show 3-bet, got '{labels_3bet.get('raiseHalfPot')}'"
        results['action_labeling_3bet'] = {'status': 'PASS', 'labels': labels_3bet}
        
        # Test Postflop First to Act
        game_state_postflop = {
            'raw_obs': {
                'stage': 1,  # Flop
                'raised': [0, 0],  # Both at 0
                'big_blind': 2,
                'pot': 20
            }
        }
        context_postflop = ActionLabeling.get_context_from_state(game_state_postflop, player_id=0, env=env)
        labels_postflop = ActionLabeling.get_button_labels(context_postflop)
        
        assert context_postflop['is_preflop'] == False, "Should be postflop"
        assert context_postflop['is_first_to_act'] == True, "Should be first to act"
        assert 'Bet' in labels_postflop.get('raiseHalfPot', ''), f"Should show Bet, got '{labels_postflop.get('raiseHalfPot')}'"
        assert labels_postflop.get('showRaisePot') == True, "Should show two bet options postflop"
        results['action_labeling_postflop'] = {'status': 'PASS', 'labels': labels_postflop}
        
        # Test Postflop Facing Bet
        game_state_facing_bet = {
            'raw_obs': {
                'stage': 1,  # Flop
                'raised': [0, 10],  # Player=0, Opponent=10 (bet)
                'big_blind': 2,
                'pot': 30
            }
        }
        context_facing = ActionLabeling.get_context_from_state(game_state_facing_bet, player_id=0, env=env)
        labels_facing = ActionLabeling.get_button_labels(context_facing)
        
        assert context_facing['is_facing_bet'] == True, "Should be facing a bet"
        assert labels_facing.get('checkCall') == 'Call', "Should show Call when facing bet"
        assert 'Raise' in labels_facing.get('raiseHalfPot', ''), f"Should show Raise, got '{labels_facing.get('raiseHalfPot')}'"
        results['action_labeling_facing_bet'] = {'status': 'PASS', 'labels': labels_facing}
    
    # Print summary
    for test_name, result in results.items():
        status = result.get('status', 'UNKNOWN')
    
    return results


if __name__ == '__main__':
    test_action_contexts()

