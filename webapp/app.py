"""
Flask application for No Limit Texas Hold'em with AI Poker Coach
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback
import time
import logging
from datetime import datetime
import numpy as np
import rlcard
from rlcard.agents import RandomAgent
from coach.strategy_evaluator import StrategyEvaluator
from coach.chatbot_coach import ChatbotCoach
from coach.gto_agent import GTOAgent
from coach.llm_opponent_agent import LLMOpponentAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable werkzeug HTTP request logging (suppress lines like "GET /api/game/state HTTP/1.1" 200)
logging.getLogger('werkzeug').setLevel(logging.WARNING)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class WebHumanAgent:
    """Simple human agent for web interface that doesn't block"""
    def __init__(self, num_actions):
        self.use_raw = True
        self.num_actions = num_actions
        self.action = None
    
    def step(self, state):
        """Return the stored action"""
        if self.action is None:
            raise ValueError("No action set")
        action = self.action
        self.action = None
        return action
    
    def set_action(self, action):
        """Set the action to take"""
        self.action = action
    
    def eval_step(self, state):
        """Evaluation step - same as step for human agent"""
        return self.step(state), {}

app = Flask(__name__)
CORS(app)

# Initialize strategy evaluator
evaluator = StrategyEvaluator()

# Initialize chatbot coach (singleton) - Phase 3
try:
    chatbot_coach = ChatbotCoach()
except Exception as e:
    logger.error(f"Failed to initialize ChatbotCoach: {e}")
    chatbot_coach = None

# In-memory storage for hand history (session-based)
hand_history_storage = {}

# Game manager for managing game sessions
class GameManager:
    def __init__(self):
        self.games = {}  # session_id -> game state
    
    def start_game(self, session_id):
        """Start a new game for a session"""
        # Create environment
        env = rlcard.make('no-limit-holdem')
        
        # Create agents
        human_agent = WebHumanAgent(env.num_actions)
        # Use LLM opponent agent instead of GTO agent for LLM-powered decisions
        # Falls back to GTOAgent on LLM failures
        ai_agent = LLMOpponentAgent(num_actions=env.num_actions)
        env.set_agents([human_agent, ai_agent])
        
        # Reset environment
        state, player_id = env.reset()
        
        # Store game state
        # Get player's hand from initial state
        # Note: If it's not player 0's turn initially, we'll get the hand when it becomes player 0's turn
        initial_raw_obs = state.get('raw_obs', {})
        player_hand = []
        # Only get hand if it's player 0's turn, otherwise we'll get it later
        if player_id == 0:
            player_hand = initial_raw_obs.get('hand', [])
        
        self.games[session_id] = {
            'env': env,
            'human_agent': human_agent,
            'ai_agent': ai_agent,
            'current_state': state,
            'current_player': player_id,
            'hand_history': [],
            'last_stage': 0,  # Track previous stage for detecting community card deals
            'last_public_cards_count': 0,  # Track previous public cards count
            'action_history_with_cards': [],  # Store action history with community cards
            'last_action_count': 0,  # Track number of actions processed
            'blinds_added': False,  # Track if blind entries have been added
            'player_hand': player_hand.copy() if player_hand else []  # Store player's hand for the entire hand
        }
        
        return self.get_game_state(session_id)
    
    def get_game_state(self, session_id):
        """Get current game state for a session"""
        if session_id not in self.games:
            return None
        
        game = self.games[session_id]
        env = game['env']
        state = game['current_state']
        current_player = game['current_player']
        
        # Get raw observation for current player
        raw_obs = state['raw_obs']
        
        # IMPORTANT: The hand in raw_obs is from the current player's perspective
        # When current_player == 1, raw_obs contains the opponent's hand, not player 0's hand!
        # We need to use the stored player_hand instead of raw_obs.get('hand')
        stored_player_hand = game.get('player_hand', [])
        last_stage = game.get('last_stage', -1)
        current_stage = raw_obs.get('stage', 0)
        if hasattr(current_stage, 'value'):
            current_stage = current_stage.value
        elif not isinstance(current_stage, int):
            current_stage = int(current_stage) if current_stage else 0
        
        # Detect new hand: stage resets to 0 after being > 0 (hand ended and new one started)
        # Also detect if game was over and now we're starting fresh
        was_over = game.get('was_over', False)
        is_over = env.is_over()
        is_new_hand = (last_stage > 0 and current_stage == 0) or (was_over and not is_over and current_stage == 0)
        
        # If we're at preflop (stage 0) and it's player 0's turn, update stored hand
        # This captures the hand at the start of each new hand
        if current_stage == 0 and current_player == 0:
            current_hand = raw_obs.get('hand', [])
            if current_hand and len(current_hand) == 2:
                # Check if this is actually a new hand (different cards or new hand detected)
                if is_new_hand or not stored_player_hand or stored_player_hand != current_hand:
                    # New hand dealt - update stored hand
                    game['player_hand'] = current_hand.copy()
                    stored_player_hand = current_hand.copy()
        # If stored hand is empty but we have a hand in raw_obs and it's player 0's turn, use it
        elif not stored_player_hand and current_player == 0:
            current_hand = raw_obs.get('hand', [])
            if current_hand and len(current_hand) == 2:
                game['player_hand'] = current_hand.copy()
                stored_player_hand = current_hand.copy()
        
        # Track if game is over for next iteration
        game['was_over'] = is_over
        
        # Use stored player hand instead of raw_obs hand
        # Always use stored hand if available, fallback to raw_obs only if we're player 0 and no stored hand
        if stored_player_hand:
            player_hand = stored_player_hand
        elif current_player == 0:
            # Fallback: use raw_obs if we're player 0 and no stored hand yet
            player_hand = raw_obs.get('hand', [])
        else:
            # If it's opponent's turn, use stored hand (should be from previous player 0 turn)
            player_hand = stored_player_hand if stored_player_hand else []
        
        # Convert Action objects to integers for JSON serialization
        def convert_actions(actions):
            """Convert Action objects to integers"""
            if not actions:
                return []
            result = []
            # Handle dict/OrderedDict (legal_actions is often a dict)
            if isinstance(actions, dict):
                actions = list(actions.keys())
            for action in actions:
                if hasattr(action, 'value'):
                    # Action enum - use .value
                    result.append(action.value)
                elif isinstance(action, int):
                    result.append(action)
                else:
                    # Try to convert to int
                    try:
                        result.append(int(action))
                    except (ValueError, TypeError):
                        # If all else fails, try to get the value attribute
                        action_value = getattr(action, 'action', None)
                        if action_value is not None:
                            result.append(action_value)
                        # If still no value, skip this action (don't add 0 as fallback)
            return result
        
        legal_actions = convert_actions(raw_obs.get('legal_actions', []))
        raw_legal_actions = convert_actions(raw_obs.get('raw_legal_actions', []))
        
        # Convert stage enum to integer if needed
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        elif not isinstance(stage, int):
            stage = int(stage) if stage else 0
        
        # Get dealer_id from game object (not in raw_obs)
        dealer_id = None
        if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
            dealer_id = env.game.dealer_id
        
        # Calculate button_id (small blind in heads-up)
        button_id = None
        if dealer_id is not None:
            button_id = (dealer_id + 1) % 2  # In HUNL, button = (dealer + 1) % 2
        
        # Get in_chips from players (amount each player has bet in current hand)
        in_chips = [0, 0]
        if hasattr(env, 'game') and hasattr(env.game, 'players'):
            try:
                in_chips = [int(p.in_chips) for p in env.game.players]
            except:
                pass
        
        # Get raised array (amount bet in current betting round)
        raised = raw_obs.get('raised', [0, 0])
        
        # Build game state response
        game_state = {
            'hand': player_hand,  # Use stored player hand, not raw_obs hand
            'public_cards': raw_obs.get('public_cards', []),
            'stakes': raw_obs.get('stakes', [0, 0]),
            'pot': raw_obs.get('pot', 0),
            'stage': stage,
            'legal_actions': legal_actions,
            'raw_legal_actions': raw_legal_actions,
            'current_player': current_player,
            'is_waiting_for_action': not env.is_over() and current_player == 0,
            'is_over': env.is_over(),
            'big_blind': raw_obs.get('big_blind', 2),
            'button_id': button_id,
            'dealer_id': dealer_id,
            'raised': raised,
            'in_chips': in_chips,  # Total chips each player has put in this hand
            'action_history': self._build_action_history(env, state, session_id),
            'payoffs': env.get_payoffs() if env.is_over() else None,
            'opponent_hand': self._get_opponent_hand(env) if env.is_over() else None
        }
        
        # Convert numpy types to native Python types for JSON serialization
        game_state = convert_numpy_types(game_state)
        
        return game_state
    
    def _build_action_history(self, env, state, session_id):
        """Build action history from environment, including community card deals and blinds"""
        game = self.games.get(session_id, {})
        
        # Initialize cumulative history if not exists
        if 'action_history_with_cards' not in game:
            game['action_history_with_cards'] = []
            game['last_action_count'] = 0
            game['blinds_added'] = False
        
        try:
            # Get current stage and public cards
            raw_obs = state.get('raw_obs', {})
            current_stage = raw_obs.get('stage', 0)
            if hasattr(current_stage, 'value'):
                current_stage = current_stage.value
            elif not isinstance(current_stage, int):
                current_stage = int(current_stage) if current_stage else 0
            
            public_cards = raw_obs.get('public_cards', [])
            current_public_cards_count = len(public_cards)
            
            # Get previous state from game storage
            last_stage = game.get('last_stage', 0)
            last_public_cards_count = game.get('last_public_cards_count', 0)
            
            # FIRST: Check if blinds need to be added (at start of preflop, before any actions)
            if not game.get('blinds_added', False) and current_stage == 0:
                # Get blind information
                big_blind = raw_obs.get('big_blind', 2)
                small_blind = big_blind // 2
                
                # Get dealer_id to determine who posted which blind
                dealer_id = None
                if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                    dealer_id = env.game.dealer_id
                
                # Try to get player in_chips directly from game object (more reliable)
                player_in_chips = None
                if hasattr(env, 'game') and hasattr(env.game, 'players'):
                    try:
                        player_in_chips = [p.in_chips for p in env.game.players]
                    except:
                        pass
                
                # Fallback to raised array if player_in_chips not available
                if player_in_chips is None:
                    player_in_chips = raw_obs.get('raised', [0, 0])
                
                # In heads-up: small blind = (dealer + 1) % 2, big blind = (dealer + 2) % 2 = dealer
                if dealer_id is not None and len(player_in_chips) >= 2:
                    small_blind_player = (dealer_id + 1) % 2
                    big_blind_player = (dealer_id + 2) % 2  # In 2-player, this wraps to dealer_id
                    
                    # Check if blinds have actually been posted
                    # Check both small blind and big blind amounts
                    small_blind_posted = player_in_chips[small_blind_player] >= small_blind - 0.5
                    big_blind_posted = player_in_chips[big_blind_player] >= big_blind - 0.5
                    
                    if small_blind_posted and big_blind_posted:
                        # Add small blind entry
                        game['action_history_with_cards'].append({
                            'type': 'blind',
                            'blind_type': 'small',
                            'player_id': small_blind_player,
                            'player_name': 'You' if small_blind_player == 0 else 'Opponent',
                            'amount': small_blind
                        })
                        
                        # Add big blind entry
                        game['action_history_with_cards'].append({
                            'type': 'blind',
                            'blind_type': 'big',
                            'player_id': big_blind_player,
                            'player_name': 'You' if big_blind_player == 0 else 'Opponent',
                            'amount': big_blind
                        })
                        
                        # Log blinds posted
                        logger.info(f"💰 Blinds posted - Session: {session_id}, SB: {small_blind} (P{small_blind_player}), BB: {big_blind} (P{big_blind_player}), Dealer: {dealer_id}")
                        
                        game['blinds_added'] = True
            
            # SECOND: Process all actions from action_recorder FIRST (before community cards)
            # This ensures actions appear in correct chronological order
            current_action_count = 0
            if hasattr(env, 'action_recorder') and env.action_recorder:
                current_action_count = len(env.action_recorder)
                
                # Add new actions to cumulative history
                last_action_count = game.get('last_action_count', 0)
                if current_action_count > last_action_count:
                    for i in range(last_action_count, current_action_count):
                        action_record = env.action_recorder[i]
                        if len(action_record) >= 2:
                            player_id, action = action_record[0], action_record[1]
                            # Reconstruct the state BEFORE this action was taken
                            # This ensures action labels are correct based on the context at that time
                            state_before_action = self._reconstruct_state_before_action(
                                env, state, env.action_recorder[:i], i
                            )
                            # Reconstruct the state AFTER this action to get actual bet amount
                            state_after_action = self._reconstruct_state_after_action(
                                env, state, env.action_recorder[:i+1], i+1
                            )
                            # Get bet amount from state after action (more accurate)
                            bet_amount = self._get_bet_amount_from_state(action, env, player_id, state_after_action)
                            
                            # If bet_amount is still 0, try calculating it directly
                            if bet_amount == 0:
                                temp_state_before = state_before_action
                                bet_amount = self._get_bet_amount(action, env, player_id, temp_state_before)
                            
                            # Format action name, using bet_amount to help determine correct label
                            action_name = self._format_action_with_context(
                                action, env, state_before_action, player_id, env.action_recorder[:i], bet_amount
                            )
                            game['action_history_with_cards'].append({
                                'type': 'action',
                                'player_id': player_id,
                                'player_name': 'You' if player_id == 0 else 'Opponent',
                                'action': action_name,
                                'bet_amount': bet_amount
                            })
                            
                            # Log action added to history
                            logger.info(f"🎯 Action added - Session: {session_id}, P{player_id} ({'You' if player_id == 0 else 'Opp'}), {action_name}, Bet: {bet_amount}, Stage: {current_stage}")
                
                game['last_action_count'] = current_action_count
            
            # THIRD: Check if community cards were dealt (AFTER processing actions)
            # This ensures community cards appear after all actions in the previous street
            if current_public_cards_count > last_public_cards_count:
                # Determine which cards were just dealt
                new_cards = public_cards[last_public_cards_count:]
                
                # Determine stage name
                stage_name = 'Preflop'
                if current_stage == 1:
                    stage_name = 'Flop'
                elif current_stage == 2:
                    stage_name = 'Turn'
                elif current_stage == 3:
                    stage_name = 'River'
                
                # Add community card entry to cumulative history
                game['action_history_with_cards'].append({
                    'type': 'community_cards',
                    'stage': stage_name,
                    'cards': new_cards,
                    'all_cards': public_cards.copy()  # Include all cards for display
                })
                
                # Log community cards dealt
                logger.info(f"🃏 Cards dealt - Session: {session_id}, {stage_name}, New: {new_cards}, Total: {len(public_cards)}")
            
            # Update stored state
            game['last_stage'] = current_stage
            game['last_public_cards_count'] = current_public_cards_count
            
            # Return the cumulative history
            return game['action_history_with_cards'].copy()
            
        except Exception as e:
            # If we can't build action history, return what we have
            print(f"Warning: Could not build action history: {e}")
            return game.get('action_history_with_cards', []).copy()
    
    def _format_action(self, action, env, state=None, player_id=None):
        """Format action for display"""
        return self._action_to_string(action, env, state, player_id)
    
    def _reconstruct_state_before_action(self, env, state, previous_actions, action_index):
        """
        Reconstruct the game state before a specific action was taken.
        This is needed to correctly format action labels based on the context at that time.
        """
        # Start with the current state's raw_obs
        raw_obs = state.get('raw_obs', {}).copy()
        
        # Get initial values
        big_blind = raw_obs.get('big_blind', 2)
        small_blind = big_blind // 2
        
        # Get current stage to determine if we're postflop
        current_stage = raw_obs.get('stage', 0)
        if hasattr(current_stage, 'value'):
            current_stage = current_stage.value
        elif not isinstance(current_stage, int):
            current_stage = int(current_stage) if current_stage else 0
        
        # Get public cards count to determine betting round
        public_cards = raw_obs.get('public_cards', [])
        current_public_cards_count = len(public_cards)
        
        # Determine which betting round this action belongs to
        # Preflop: 0 cards, Flop: 3 cards, Turn: 4 cards, River: 5 cards
        # We need to find when the current betting round started
        # by tracking when public cards were added
        
        # Initialize raised amounts
        # For postflop actions, raised should start at [0, 0] for the new betting round
        # For preflop actions, raised starts with blinds
        dealer_id = None
        if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
            dealer_id = env.game.dealer_id
        
        raised = [0, 0]
        
        # Determine which betting round this action belongs to
        # Preflop: 0 cards, Flop: 3 cards, Turn: 4 cards, River: 5 cards
        action_stage = 0  # Default to preflop
        if current_public_cards_count >= 5:
            action_stage = 3  # River
        elif current_public_cards_count >= 4:
            action_stage = 2  # Turn
        elif current_public_cards_count >= 3:
            action_stage = 1  # Flop
        else:
            action_stage = 0  # Preflop
        
        # For postflop actions, raised should start at [0, 0] for the new betting round
        # The raised array is reset when community cards are dealt
        if action_stage > 0:  # Postflop
            # Start with raised = [0, 0] for postflop betting rounds
            # We'll replay actions, but the check/call logic in _action_to_string
            # will handle the case where raised amounts are equal (it's a check)
            raised = [0, 0]
            betting_round_start_index = 0
        else:
            # Preflop: initialize with blinds
            if dealer_id is not None:
                small_blind_player = (dealer_id + 1) % 2
                big_blind_player = (dealer_id + 2) % 2
                raised[small_blind_player] = small_blind
                raised[big_blind_player] = big_blind
            betting_round_start_index = 0
        
        # Replay all previous actions to get raised amounts before this action
        # previous_actions contains actions up to but not including the current action
        # action_index is the index of the current action (so we process indices 0 to action_index-1)
        if previous_actions and action_index > 0:
            # Process all actions in previous_actions (they are indices 0 to action_index-1)
            for i in range(len(previous_actions)):
                prev_record = previous_actions[i]
                if len(prev_record) >= 2:
                    prev_player_id, prev_action = prev_record[0], prev_record[1]
                    
                    # Get action value
                    if hasattr(prev_action, 'value'):
                        prev_action_value = prev_action.value
                    elif isinstance(prev_action, int):
                        prev_action_value = prev_action
                    else:
                        try:
                            prev_action_value = int(prev_action)
                        except:
                            continue
                    
                    # Reconstruct state at this point to calculate bet amount
                    temp_state = {
                        'raw_obs': {
                            'raised': raised.copy(),
                            'big_blind': big_blind,
                            'pot': sum(raised),
                            'stage': raw_obs.get('stage', 0)
                        }
                    }
                    bet_amount = self._get_bet_amount(prev_action, env, prev_player_id, temp_state)
                    
                    # Update raised amounts based on action type
                    if prev_action_value in [2, 3]:  # Raise actions
                        # For raises, the raised amount becomes the total bet amount
                        # bet_amount is the total chips the player is betting TO (not additional)
                        opponent_id = 1 - prev_player_id
                        opponent_raised = raised[opponent_id]
                        # The new raised amount is the total amount they're betting to
                        # Make sure bet_amount is at least as much as opponent_raised (minimum call)
                        if bet_amount < opponent_raised:
                            # If bet_amount is less than opponent_raised, it's likely wrong
                            # Use opponent_raised + minimum raise instead
                            raised[prev_player_id] = opponent_raised + big_blind
                        else:
                            raised[prev_player_id] = bet_amount
                    elif prev_action_value == 1:  # Call/Check
                        # For calls, match the opponent's raised amount
                        opponent_id = 1 - prev_player_id
                        raised[prev_player_id] = raised[opponent_id]
                    # Fold (0) and All-in (4) don't change raised amounts for the other player
        
        # Create reconstructed state
        reconstructed_state = state.copy()
        reconstructed_state['raw_obs'] = raw_obs.copy()
        reconstructed_state['raw_obs']['raised'] = raised
        # Recalculate pot based on raised amounts
        reconstructed_state['raw_obs']['pot'] = sum(raised)

        return reconstructed_state
    
    def _reconstruct_state_after_action(self, env, state, previous_actions, action_index):
        """
        Reconstruct the game state after a specific action was taken.
        This is used to get the actual bet amount after the action.
        """
        if not previous_actions or action_index > len(previous_actions):
            return state
        
        # Start with the current state's raw_obs
        raw_obs = state.get('raw_obs', {}).copy()
        
        # Get initial values
        big_blind = raw_obs.get('big_blind', 2)
        small_blind = big_blind // 2
        
        # Initialize raised amounts based on blinds
        dealer_id = None
        if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
            dealer_id = env.game.dealer_id
        
        raised = [0, 0]
        if dealer_id is not None:
            small_blind_player = (dealer_id + 1) % 2
            big_blind_player = (dealer_id + 2) % 2
            raised[small_blind_player] = small_blind
            raised[big_blind_player] = big_blind
        
        # Replay all actions up to and including this action
        for i in range(action_index):
            if i < len(previous_actions):
                prev_record = previous_actions[i]
                if len(prev_record) >= 2:
                    prev_player_id, prev_action = prev_record[0], prev_record[1]
                    
                    # Get action value
                    if hasattr(prev_action, 'value'):
                        prev_action_value = prev_action.value
                    elif isinstance(prev_action, int):
                        prev_action_value = prev_action
                    else:
                        try:
                            prev_action_value = int(prev_action)
                        except:
                            continue
                    
                    # Reconstruct state at this point to calculate bet amount
                    temp_state = {
                        'raw_obs': {
                            'raised': raised.copy(),
                            'big_blind': big_blind,
                            'pot': sum(raised),
                            'stage': raw_obs.get('stage', 0)
                        }
                    }
                    bet_amount = self._get_bet_amount(prev_action, env, prev_player_id, temp_state)
                    
                    # Update raised amounts based on action type
                    if prev_action_value in [2, 3]:  # Raise actions
                        opponent_id = 1 - prev_player_id
                        opponent_raised = raised[opponent_id]
                        if bet_amount < opponent_raised:
                            raised[prev_player_id] = opponent_raised + big_blind
                        else:
                            raised[prev_player_id] = bet_amount
                    elif prev_action_value == 1:  # Call/Check
                        opponent_id = 1 - prev_player_id
                        # Only call if opponent has raised more than player (facing a bet)
                        # If raised amounts are equal, it's a check - don't change raised amount
                        if raised[opponent_id] > raised[prev_player_id]:
                            raised[prev_player_id] = raised[opponent_id]
        
        # Create reconstructed state
        reconstructed_state = state.copy()
        reconstructed_state['raw_obs'] = raw_obs.copy()
        reconstructed_state['raw_obs']['raised'] = raised
        reconstructed_state['raw_obs']['pot'] = sum(raised)
        
        return reconstructed_state
    
    def _get_bet_amount_from_state(self, action, env, player_id, state):
        """Get bet amount from state's raised array (after action)"""
        if state and 'raw_obs' in state:
            raw_obs = state['raw_obs']
            raised = raw_obs.get('raised', [0, 0])
            if player_id is not None and player_id < len(raised):
                return raised[player_id]
        # Fallback to regular bet amount calculation
        return self._get_bet_amount(action, env, player_id, state)
    
    def _format_action_with_context(self, action, env, state=None, player_id=None, previous_actions=None, bet_amount=None):
        """Format action for display with context from previous actions"""
        # Check if this is a Check/Call action (action_value == 1)
        action_value = action
        if hasattr(action, 'value'):
            action_value = action.value
        elif not isinstance(action, int):
            try:
                action_value = int(action)
            except:
                pass
        
        # Only need special handling for Check/Call actions in specific cases
        # Don't override the reconstructed state if it already correctly shows the current betting round state
        if action_value == 1 and state and 'raw_obs' in state:
            raw_obs = state['raw_obs']
            current_raised = raw_obs.get('raised', [0, 0])
            player_raised = current_raised[player_id] if player_id is not None and player_id < len(current_raised) else 0
            opponent_raised = current_raised[1 - player_id] if player_id is not None and len(current_raised) > 1 - player_id else 0

            # If the reconstructed state already shows opponent_raised > player_raised, use it as-is
            # Only apply special logic if the reconstructed state shows equal raised amounts but we know opponent raised recently
            if abs(opponent_raised - player_raised) <= 0.01 and previous_actions is not None and len(previous_actions) > 0:
                # Check if opponent raised in the SAME betting round (not across betting rounds)
                # This is only needed when state reconstruction doesn't properly reflect the current betting round
                opponent_raised_same_round = False
                stage = raw_obs.get('stage', 0)
                if hasattr(stage, 'value'):
                    stage = stage.value
                elif not isinstance(stage, int):
                    stage = int(stage) if stage else 0

                # For postflop, if raised is [0,0], it means no one raised in this betting round
                # Don't override based on previous betting rounds
                if stage == 0:  # Only check previous actions for preflop
                    for i in range(len(previous_actions) - 1, -1, -1):
                        prev_action_record = previous_actions[i]
                        if len(prev_action_record) >= 2:
                            prev_player_id, prev_action = prev_action_record[0], prev_action_record[1]
                            if prev_player_id == 1 - player_id:  # Opponent's action
                                # Get action value
                                if hasattr(prev_action, 'value'):
                                    prev_action_value = prev_action.value
                                elif isinstance(prev_action, int):
                                    prev_action_value = prev_action
                                else:
                                    try:
                                        prev_action_value = int(prev_action)
                                    except:
                                        continue
                                # If opponent raised (action 2 or 3), player is facing a bet
                                if prev_action_value in [2, 3]:  # Raise actions
                                    opponent_raised_same_round = True
                                # Stop at the most recent opponent action
                                break

                    # If opponent raised in same round, create modified state to indicate player is facing a bet
                    if opponent_raised_same_round:
                        modified_state = state.copy()
                        if 'raw_obs' not in modified_state:
                            modified_state['raw_obs'] = {}
                        elif not isinstance(modified_state['raw_obs'], dict):
                            modified_state['raw_obs'] = dict(modified_state['raw_obs'])
                        else:
                            modified_state['raw_obs'] = modified_state['raw_obs'].copy()

                        # Ensure opponent_raised > player_raised to trigger "Call"
                        modified_state['raw_obs']['raised'] = current_raised.copy()
                        modified_state['raw_obs']['raised'][1 - player_id] = player_raised + 1

                        return self._action_to_string(action, env, modified_state, player_id, bet_amount)
        
        # Fall back to regular formatting
        result = self._action_to_string(action, env, state, player_id, bet_amount)
        return result
    
    def _action_to_string(self, action, env=None, state=None, player_id=None, bet_amount=None):
        """Convert action value to display string"""
        # Handle Action enum objects (from RLCard)
        if hasattr(action, 'value'):
            action_value = action.value
        elif isinstance(action, int):
            action_value = action
        else:
            # Try to convert to int
            try:
                action_value = int(action)
            except (ValueError, TypeError):
                return f'Action {action}'
        
        # Map action values to display names
        if action_value == 0:
            return 'Fold'
        elif action_value == 1:
            # Determine if it's "Check" or "Call" based on bet_amount (simplest and most reliable)
            # If bet_amount is provided and > 0, it's a Call; if 0 or None, it's a Check
            if bet_amount is not None and bet_amount > 0:
                return 'Call'
            elif bet_amount is not None and bet_amount == 0:
                return 'Check'
            
            # Fallback: Determine if it's "Check" or "Call" based on raised amounts
            if env is not None and state is not None and player_id is not None:
                try:
                    raw_obs = state.get('raw_obs', {})
                    raised = raw_obs.get('raised', [0, 0])
                    stage = raw_obs.get('stage', 0)
                    if hasattr(stage, 'value'):
                        stage = stage.value
                    elif not isinstance(stage, int):
                        stage = int(stage) if stage else 0
                    
                    big_blind = raw_obs.get('big_blind', 2)
                    player_raised = raised[player_id] if player_id < len(raised) else 0
                    opponent_raised = raised[1 - player_id] if len(raised) > 1 - player_id else 0
                    
                    # Preflop: Small blind can "Check" (complete blind) if big blind hasn't raised
                    # Postflop: "Check" if player has matched opponent's bet, "Call" if opponent has bet more
                    is_preflop = stage == 0
                    
                    # Get dealer_id to determine button
                    dealer_id = None
                    is_small_blind = False
                    if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                        dealer_id = env.game.dealer_id
                        if dealer_id is not None:
                            button_id = (dealer_id + 1) % 2
                            is_small_blind = button_id == player_id
                    
                    # Determine if facing a bet
                    # Facing a bet if opponent has bet more than player (regardless of preflop/postflop)
                    # The small blind posting 0.5BB and big blind posting 1.0BB means small blind IS facing a bet
                    # Use a small epsilon to handle floating point comparison issues
                    epsilon = 0.01
                    
                    # Special case for postflop: if both players have equal raised amounts (or both are 0),
                    # it's a check, not a call. This handles the case where raised array might have stale
                    # values from preflop when reconstructing state for postflop actions.
                    if not is_preflop:
                        # For postflop, if raised amounts are equal (within epsilon), it's a check
                        if abs(opponent_raised - player_raised) <= epsilon:
                            is_facing_bet = False
                        else:
                            is_facing_bet = (opponent_raised - player_raised) > epsilon
                    else:
                        is_facing_bet = (opponent_raised - player_raised) > epsilon
                    
                    # Additional check: look at action history to see if opponent just raised
                    # This helps when the raised array might not be perfectly synchronized
                    if not is_facing_bet and hasattr(env, 'action_recorder') and env.action_recorder:
                        # Check recent actions to see if opponent raised
                        for i in range(len(env.action_recorder) - 1, max(-1, len(env.action_recorder) - 5), -1):
                            if i >= 0 and len(env.action_recorder[i]) >= 2:
                                action_player_id, action_val = env.action_recorder[i][0], env.action_recorder[i][1]
                                # If opponent (not the current player) just raised, we're facing a bet
                                if action_player_id == 1 - player_id:
                                    # Check if it was a raise action (value 2 or 3)
                                    if action_val in [2, 3] or (hasattr(action_val, 'value') and action_val.value in [2, 3]):
                                        is_facing_bet = True
                                        break
                                    # If opponent called or checked, stop checking (only if most recent)
                                    if (action_val == 1 or (hasattr(action_val, 'value') and action_val.value == 1)) and i == len(env.action_recorder) - 1:
                                        break
                    
                    # Special case: Preflop small blind always faces a bet from big blind
                    # Even if raised array shows equal amounts after both act, small blind initially faces 1BB vs 0.5BB
                    if is_preflop and is_small_blind and not is_facing_bet:
                        # Check if we're at the start of preflop (small blind's first action)
                        # If opponent has posted the big blind (1BB) and we've only posted small blind (0.5BB)
                        # then we're facing a bet
                        if opponent_raised >= big_blind * 0.9 and player_raised < big_blind * 0.9:
                            is_facing_bet = True
                    
                    # If facing a bet, it's always "Call" (never "Check")
                    if is_facing_bet:
                        return 'Call'
                    
                    # If not facing a bet, it's "Check"
                    return 'Check'
                except Exception as e:
                    # Fallback to "Check/Call" if we can't determine
                    logger.debug(f"Could not determine Check vs Call: {e}")
                    pass
            return 'Check/Call'
        elif action_value == 2:
            # Determine appropriate label based on context
            if env is not None and state is not None and player_id is not None:
                try:
                    raw_obs = state.get('raw_obs', {})
                    stage = raw_obs.get('stage', 0)
                    if hasattr(stage, 'value'):
                        stage = stage.value
                    elif not isinstance(stage, int):
                        stage = int(stage) if stage else 0
                    
                    raised = raw_obs.get('raised', [0, 0])
                    pot = raw_obs.get('pot', 0)
                    big_blind = raw_obs.get('big_blind', 2)
                    player_raised = raised[player_id] if player_id < len(raised) else 0
                    opponent_raised = raised[1 - player_id] if len(raised) > 1 - player_id else 0
                    
                    # Check if preflop
                    if stage == 0:
                        # Get dealer_id to determine button
                        dealer_id = None
                        if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                            dealer_id = env.game.dealer_id
                        
                        # In heads-up, small blind = (dealer + 1) % 2
                        if dealer_id is not None:
                            button_id = (dealer_id + 1) % 2
                            is_small_blind = button_id == player_id
                            is_first_to_act = player_raised == opponent_raised and player_raised <= big_blind
                            is_facing_bet = opponent_raised > player_raised
                            
                            # Determine betting level based on opponent's raise amount, not pot size
                            opponent_raised_bb = opponent_raised / big_blind if big_blind > 0 else 0
                            
                            # IMPORTANT: Check is_facing_bet FIRST - if facing a bet, we're not "first to act" for betting purposes
                            if is_facing_bet:
                                # Use bet_amount if available to determine the actual raise size
                                # This helps when the state reconstruction might not be perfect
                                if bet_amount is not None and bet_amount > 0:
                                    bet_amount_bb = bet_amount / big_blind if big_blind > 0 else 0
                                    # If bet_amount is 10BB, this is a 3-bet
                                    if 9.5 <= bet_amount_bb <= 10.5:
                                        return '3-bet to 10BB'
                                    # If bet_amount is 15BB, this is a 4-bet
                                    elif 14.5 <= bet_amount_bb <= 15.5:
                                        return '4-bet to 15BB'
                                    # If bet_amount is 18BB, this is a 4-bet (pot raise)
                                    elif 17.5 <= bet_amount_bb <= 18.5:
                                        return '4-bet to 18BB'
                                    # If bet_amount is 7-8BB, this is also a 3-bet (smaller sizing)
                                    elif 6.5 <= bet_amount_bb <= 8.5:
                                        return f'3-bet to {int(round(bet_amount_bb))}BB'

                                # Fallback: Determine what we're facing based on opponent's raise
                                if opponent_raised_bb <= 1.1:
                                    # Opponent just posted big blind (or checked) - shouldn't happen for RAISE_HALF_POT
                                    return 'Raise to 3BB'
                                elif opponent_raised_bb <= 4.0:
                                    # Facing a button open (3BB) - this is a 3-bet, should be to 10BB
                                    return '3-bet to 10BB'
                                elif opponent_raised_bb <= 12.0:
                                    # Facing a 3-bet (7-10BB) - this is a 4-bet
                                    return '4-bet to 15BB'
                                else:
                                    # Facing a 4-bet or higher, calculate raise sizes in BB
                                    bet_to_call = opponent_raised - player_raised
                                    raise_25x_total = bet_to_call * 2.5 + bet_to_call
                                    raise_25x_bb = round(raise_25x_total / big_blind) if big_blind > 0 else 0
                                    return f'Raise to {raise_25x_bb}BB'
                            elif is_small_blind and is_first_to_act:
                                # Small blind opening (unopened pot, only blinds posted)
                                # Only show this if NOT facing a bet
                                return 'Raise to 3BB'
                            else:
                                # Big blind not facing a bet (can still raise) - show BB amounts
                                # Standard preflop raise sizes: 3BB and 4BB
                                return 'Raise to 3BB'
                    else:
                        # Postflop
                        is_first_to_act = player_raised == opponent_raised
                        is_facing_bet = opponent_raised > player_raised
                        
                        if is_first_to_act:
                            return 'Bet ½ Pot'
                        elif is_facing_bet:
                            bet_to_call = opponent_raised - player_raised
                            raise_25x_total = bet_to_call * 2.5 + bet_to_call
                            raise_25x_bb = round(raise_25x_total / big_blind) if big_blind > 0 else 0
                            return f'Raise to {raise_25x_bb}BB'
                except Exception:
                    # Fallback to normal raise text if we can't determine
                    pass
            return 'Raise ½ Pot'
        elif action_value == 3:
            # Determine appropriate label based on context
            if env is not None and state is not None and player_id is not None:
                try:
                    raw_obs = state.get('raw_obs', {})
                    stage = raw_obs.get('stage', 0)
                    if hasattr(stage, 'value'):
                        stage = stage.value
                    elif not isinstance(stage, int):
                        stage = int(stage) if stage else 0
                    
                    raised = raw_obs.get('raised', [0, 0])
                    pot = raw_obs.get('pot', 0)
                    big_blind = raw_obs.get('big_blind', 2)
                    player_raised = raised[player_id] if player_id < len(raised) else 0
                    opponent_raised = raised[1 - player_id] if len(raised) > 1 - player_id else 0
                    
                    # Check if preflop
                    if stage == 0:
                        is_facing_bet = opponent_raised > player_raised
                        
                        # Determine betting level based on opponent's raise amount, not pot size
                        # This is more accurate for preflop situations
                        opponent_raised_bb = opponent_raised / big_blind if big_blind > 0 else 0
                        player_raised_bb = player_raised / big_blind if big_blind > 0 else 0
                        
                        if is_facing_bet:
                            # Use bet_amount if available to determine the actual raise size
                            # This helps when the state reconstruction might not be perfect
                            if bet_amount is not None and bet_amount > 0:
                                bet_amount_bb = bet_amount / big_blind if big_blind > 0 else 0
                                # If bet_amount is 10BB, this is a 3-bet
                                if 9.5 <= bet_amount_bb <= 10.5:
                                    return '3-bet to 10BB'
                                # If bet_amount is 15BB, this is a 4-bet
                                elif 14.5 <= bet_amount_bb <= 15.5:
                                    return '4-bet to 15BB'
                                # If bet_amount is 18BB, this is a 4-bet (pot raise)
                                elif 17.5 <= bet_amount_bb <= 18.5:
                                    return '4-bet to 18BB'
                                # If bet_amount is 7-8BB, this is also a 3-bet (smaller sizing)
                                elif 6.5 <= bet_amount_bb <= 8.5:
                                    return f'3-bet to {int(round(bet_amount_bb))}BB'
                            
                            # Fallback: Determine what we're facing based on opponent's raise
                            if opponent_raised_bb <= 1.1:
                                # Opponent just posted big blind (or checked) - shouldn't happen for RAISE_POT
                                return 'Raise to 3BB'
                            elif opponent_raised_bb <= 4.0:
                                # Facing a button open (3BB) - this is a 3-bet, should be to 10BB
                                return '3-bet to 10BB'
                            elif opponent_raised_bb <= 12.0:
                                # Facing a 3-bet (7-10BB) - this is a 4-bet
                                return '4-bet to 18BB'
                            else:
                                # Facing a 4-bet or higher, calculate raise sizes in BB
                                bet_to_call = opponent_raised - player_raised
                                raise_3x_total = bet_to_call * 3 + bet_to_call
                                raise_3x_bb = round(raise_3x_total / big_blind) if big_blind > 0 else 0
                                return f'Raise to {raise_3x_bb}BB'
                        else:
                            # Big blind not facing a bet (can still raise) - show BB amounts
                            # Standard preflop raise size: 3BB only
                            return 'Raise to 3BB'
                    else:
                        # Postflop
                        is_first_to_act = player_raised == opponent_raised
                        is_facing_bet = opponent_raised > player_raised
                        
                        if is_first_to_act:
                            return 'Bet ⅔ Pot'
                        elif is_facing_bet:
                            bet_to_call = opponent_raised - player_raised
                            raise_3x_total = bet_to_call * 3 + bet_to_call
                            raise_3x_bb = round(raise_3x_total / big_blind) if big_blind > 0 else 0
                            return f'Raise to {raise_3x_bb}BB'
                except Exception:
                    # Fallback to normal raise text if we can't determine
                    pass
            return 'Raise Pot'
        elif action_value == 4:
            return 'All-In'
        else:
            return f'Action {action_value}'
    
    def _get_bet_amount(self, action, env, player_id=None, state=None):
        """Get bet amount for an action"""
        # Handle Action enum objects (from RLCard)
        if hasattr(action, 'value'):
            action_value = action.value
        elif isinstance(action, int):
            action_value = action
        else:
            try:
                action_value = int(action)
            except (ValueError, TypeError):
                return 0
        
        # Handle All-In action (action_value == 4)
        if action_value == 4:
            # For all-in, get the amount the player bet
            try:
                if state and 'raw_obs' in state:
                    raw_obs = state['raw_obs']
                    # The raised array shows how much each player has bet in current round
                    raised = raw_obs.get('raised', [0, 0])
                    if player_id is not None and player_id < len(raised) and raised[player_id] > 0:
                        # For all-in, the raised amount is what they bet this round
                        # This should be their remaining stack at the time of all-in
                        return raised[player_id]
                    # If raised is 0 or not available, try using in_chips
                    # Note: in_chips is cumulative for the hand, so it includes previous bets
                    # But it's better than nothing
                    in_chips = raw_obs.get('in_chips', [0, 0])
                    if player_id is not None and player_id < len(in_chips) and in_chips[player_id] > 0:
                        return in_chips[player_id]
                # Fallback: try to get from player object directly
                if hasattr(env, 'game') and hasattr(env.game, 'players') and player_id is not None:
                    if player_id < len(env.game.players):
                        player = env.game.players[player_id]
                        # Use in_chips as the all-in amount (what they've put in total)
                        if hasattr(player, 'in_chips') and player.in_chips > 0:
                            return int(player.in_chips)
            except Exception:
                # If we can't calculate, return 0 and let frontend handle it
                pass
            return 0
        
        # Get pot size from environment
        pot = 0
        try:
            # Try to get pot from dealer
            if hasattr(env, 'game') and hasattr(env.game, 'dealer') and hasattr(env.game.dealer, 'pot'):
                pot = env.game.dealer.pot
            # Fallback: try to get from state
            elif state and 'raw_obs' in state:
                pot = state['raw_obs'].get('pot', 0)
            elif hasattr(env, 'get_state'):
                temp_state = env.get_state(0)
                if temp_state and 'raw_obs' in temp_state:
                    pot = temp_state['raw_obs'].get('pot', 0)
        except Exception:
            pass
        
        # Calculate bet amount based on action type
        if action_value == 1:  # Check/Call
            # Determine if this is a check (bet amount 0) or call (bet amount to match opponent)
            try:
                if state and 'raw_obs' in state:
                    raw_obs = state['raw_obs']
                    raised = raw_obs.get('raised', [0, 0])
                    stage = raw_obs.get('stage', 0)
                    if hasattr(stage, 'value'):
                        stage = stage.value
                    elif not isinstance(stage, int):
                        stage = int(stage) if stage else 0

                    big_blind = raw_obs.get('big_blind', 2)
                    player_raised = raised[player_id] if player_id is not None and player_id < len(raised) else 0
                    opponent_raised = raised[1 - player_id] if player_id is not None and len(raised) > 1 - player_id else 0

                    # Determine if facing a bet
                    # Preflop: Small blind always faces big blind
                    # Postflop: Face bet if opponent has raised more than player
                    is_preflop = stage == 0
                    epsilon = 0.01

                    # Get dealer_id to determine positions
                    dealer_id = None
                    is_small_blind = False
                    if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                        dealer_id = env.game.dealer_id
                        if dealer_id is not None and player_id is not None:
                            button_id = (dealer_id + 1) % 2
                            is_small_blind = button_id == player_id

                    # Special case: Preflop small blind always faces a bet from big blind
                    if is_preflop and is_small_blind and opponent_raised >= big_blind * 0.9 and player_raised < big_blind * 0.9:
                        is_facing_bet = True
                    else:
                        # Standard logic: face bet if opponent has raised more
                        is_facing_bet = (opponent_raised - player_raised) > epsilon

                    if is_facing_bet:
                        # Call: return amount needed to match opponent's bet
                        return opponent_raised - player_raised
                    else:
                        # Check: return 0
                        return 0
            except Exception:
                # If we can't determine, default to 0 (check)
                pass
            return 0
        elif action_value == 2:  # Raise ½ Pot
            # Special handling for preflop raises
            # In preflop, "Raise to 3 BB" should result in 3 BB total, not half pot
            try:
                if state and 'raw_obs' in state:
                    raw_obs = state['raw_obs']
                    stage = raw_obs.get('stage', 0)
                    if hasattr(stage, 'value'):
                        stage = stage.value
                    elif not isinstance(stage, int):
                        stage = int(stage) if stage else 0
                    
                    # Check if we're in preflop (stage == 0)
                    if stage == 0:
                        big_blind = raw_obs.get('big_blind', 2)
                        raised = raw_obs.get('raised', [0, 0])
                        pot = raw_obs.get('pot', pot)  # Use pot from state if available
                        
                        # Use stage to detect preflop (more reliable than pot size)
                        # Pot can be larger after raises, so we use stage instead
                        if player_id is not None and player_id < len(raised):
                            player_raised = raised[player_id]
                            opponent_raised = max([r for i, r in enumerate(raised) if i != player_id], default=0)
                            
                            # If not facing a bet (preflop open), calculate to 3 BB total
                            if opponent_raised <= big_blind * 1.1:
                                # Preflop open - return 3 BB total (what the player raised to)
                                return 3 * big_blind
                            else:
                                # Facing a preflop raise (3BB open) - 3bet to 10BB total
                                return 10 * big_blind
            except Exception:
                pass
            
            # Fallback to standard calculation
            return int(pot / 2) if pot > 0 else 0
        elif action_value == 3:  # Raise Pot
            # Special handling for preflop 3bet
            try:
                if state and 'raw_obs' in state:
                    raw_obs = state['raw_obs']
                    stage = raw_obs.get('stage', 0)
                    if hasattr(stage, 'value'):
                        stage = stage.value
                    elif not isinstance(stage, int):
                        stage = int(stage) if stage else 0
                    
                    # Check if we're in preflop (stage == 0)
                    if stage == 0:
                        big_blind = raw_obs.get('big_blind', 2)
                        raised = raw_obs.get('raised', [0, 0])
                        pot = raw_obs.get('pot', pot)
                        
                        # Check if we're in preflop by checking stage (more reliable than pot size)
                        # Pot can be larger after raises, so we use stage instead
                        if player_id is not None and player_id < len(raised):
                            player_raised = raised[player_id]
                            opponent_raised = max([r for i, r in enumerate(raised) if i != player_id], default=0)
                            
                            # If not facing a bet (preflop open), raise to 3BB total
                            if opponent_raised <= big_blind * 1.1:
                                return 3 * big_blind
                            # If facing a preflop raise (3BB open), 3bet to 10BB total
                            elif opponent_raised > big_blind * 1.1:
                                return 10 * big_blind
            except Exception:
                pass
            
            return pot
        else:
            return 0
    
    def _get_opponent_hand(self, env):
        """Get opponent's hand when game is over"""
        if env.is_over():
            perfect_info = env.get_perfect_information()
            if 'hand_cards' in perfect_info and len(perfect_info['hand_cards']) > 1:
                return perfect_info['hand_cards'][1]
        return None
    
    def process_action(self, session_id, action_index):
        """Process a player action"""
        if session_id not in self.games:
            logger.warning(f"⚠️ Game session not found: {session_id}. Available sessions: {list(self.games.keys())}")
            return None
        
        game = self.games[session_id]
        env = game['env']
        state = game['current_state']
        current_player = game['current_player']
        human_agent = game['human_agent']
        
        # Edge case: Game already over
        if env.is_over():
            logger.warning(f"🏁 Attempted action on finished game for session {session_id}")
            return self.get_game_state(session_id)
        
        # Edge case: Not player's turn
        if current_player != 0:
            logger.warning(f"Attempted action when not player's turn for session {session_id}")
            return self.get_game_state(session_id)
        
        # Get legal actions - try raw_legal_actions first, fallback to legal_actions dict
        raw_legal_actions = state['raw_obs'].get('raw_legal_actions', [])
        legal_actions_dict = state['raw_obs'].get('legal_actions', {})
        
        # Convert legal_actions dict to list if needed
        legal_actions_list = []
        if raw_legal_actions and len(raw_legal_actions) > 0:
            legal_actions_list = raw_legal_actions
        elif legal_actions_dict:
            # Convert dict keys to list
            if isinstance(legal_actions_dict, dict):
                legal_actions_list = list(legal_actions_dict.keys())
            elif isinstance(legal_actions_dict, list):
                legal_actions_list = legal_actions_dict
        
        # Edge case: No legal actions
        if not legal_actions_list or len(legal_actions_list) == 0:
            logger.warning(f"No legal actions available for session {session_id}")
            return self.get_game_state(session_id)
        
        # Edge case: Invalid action index
        if action_index < 0 or action_index >= len(legal_actions_list):
            raise ValueError(f'Invalid action index: {action_index} (valid range: 0-{len(legal_actions_list)-1})')
        
        action = legal_actions_list[action_index]
        
        # Edge case: Action is None or invalid
        if action is None:
            raise ValueError('Action is None')
        
        # Get action details for logging
        raw_obs = state.get('raw_obs', {})
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        elif not isinstance(stage, int):
            stage = int(stage) if stage else 0
        
        stage_names = {0: 'Preflop', 1: 'Flop', 2: 'Turn', 3: 'River'}
        stage_name = stage_names.get(stage, f'Stage {stage}')
        action_name = self._action_to_string(action, env, state, current_player)
        pot = raw_obs.get('pot', 0)
        public_cards = raw_obs.get('public_cards', [])
        player_hand = game.get('player_hand', [])
        raised = raw_obs.get('raised', [0, 0])
        
        # Log player action with details
        logger.info(f"👤 Player action - Session: {session_id}, {stage_name}, {action_name} ({action}), Pot: {pot}, Cards: {len(public_cards)}, Hand: {player_hand}")
        
        # Set action in human agent
        human_agent.set_action(action)
        
        try:
            # Process action
            next_state, next_player_id = env.step(action, human_agent.use_raw)
            
            # Edge case: Environment error
            if next_state is None:
                logger.error(f"Environment returned None state for session {session_id}, action: {action}")
                # Try to return current game state as fallback
                try:
                    return self.get_game_state(session_id)
                except Exception as fallback_error:
                    logger.error(f"Failed to get game state as fallback: {fallback_error}")
                    return None
            
            # Update game state
            game['current_state'] = next_state
            game['current_player'] = next_player_id
            
            # Get updated state info for logging
            next_raw_obs = next_state.get('raw_obs', {})
            next_pot = next_raw_obs.get('pot', 0)
            next_public_cards = next_raw_obs.get('public_cards', [])
            next_raised = next_raw_obs.get('raised', [0, 0])
            
            # Log action result
            logger.info(f"✅ Action processed - Session: {session_id}, Next: P{next_player_id}, Pot: {next_pot}, Cards: {len(next_public_cards)}")
            
            # Track decision for hand history
            self._track_decision(session_id, current_player, action, state, next_state)
            
            return self.get_game_state(session_id)
        except Exception as e:
            logger.error(f"Error processing action for session {session_id}: {str(e)}")
            raise
    
    def process_ai_turn(self, session_id):
        """Process AI opponent's turn"""
        if session_id not in self.games:
            logger.warning(f"Game session not found for AI turn: {session_id}. Available sessions: {list(self.games.keys())}")
            return None
        
        game = self.games[session_id]
        env = game['env']
        state = game['current_state']
        current_player = game['current_player']
        ai_agent = game['ai_agent']
        
        # Edge case: Not AI's turn
        if current_player != 1:
            return self.get_game_state(session_id)
        
        # Edge case: Game already over
        if env.is_over():
            return self.get_game_state(session_id)
        
        try:
            # Get AI action
            action, _ = ai_agent.eval_step(state)
            
            # Edge case: AI action is None or invalid
            if action is None:
                logger.warning(f"AI agent returned None action for session {session_id}")
                # Default to fold if action is None
                legal_actions = state['raw_obs'].get('raw_legal_actions', [])
                if legal_actions and len(legal_actions) > 0:
                    action = legal_actions[0]  # First action is usually fold/check
                else:
                    return self.get_game_state(session_id)
            
            # Get action details for logging
            raw_obs = state.get('raw_obs', {})
            stage = raw_obs.get('stage', 0)
            if hasattr(stage, 'value'):
                stage = stage.value
            elif not isinstance(stage, int):
                stage = int(stage) if stage else 0
            
            stage_names = {0: 'Preflop', 1: 'Flop', 2: 'Turn', 3: 'River'}
            stage_name = stage_names.get(stage, f'Stage {stage}')
            action_name = self._action_to_string(action, env, state, current_player)
            pot = raw_obs.get('pot', 0)
            public_cards = raw_obs.get('public_cards', [])
            raised = raw_obs.get('raised', [0, 0])
            
            # Log AI action with details
            logger.info(f"🤖 AI action - Session: {session_id}, {stage_name}, {action_name} ({action}), Pot: {pot}, Cards: {len(public_cards)}")
            
            # Process action
            next_state, next_player_id = env.step(action, ai_agent.use_raw)
            
            # Edge case: Environment error
            if next_state is None:
                logger.error(f"Environment returned None state for AI turn in session {session_id}")
                return None
            
            # Update game state
            game['current_state'] = next_state
            game['current_player'] = next_player_id
            
            # Get updated state info for logging
            next_raw_obs = next_state.get('raw_obs', {})
            next_pot = next_raw_obs.get('pot', 0)
            next_public_cards = next_raw_obs.get('public_cards', [])
            next_raised = next_raw_obs.get('raised', [0, 0])
            
            # Log AI action result
            logger.info(f"🔄 AI action processed - Session: {session_id}, Next: P{next_player_id}, Pot: {next_pot}, Cards: {len(next_public_cards)}")
            
            # Track decision for hand history
            self._track_decision(session_id, current_player, action, state, next_state)
            
            return self.get_game_state(session_id)
        except Exception as e:
            logger.error(f"Error processing AI turn for session {session_id}: {str(e)}")
            # Return current state on error to prevent game from breaking
            return self.get_game_state(session_id)
    
    def _track_decision(self, session_id, player_id, action, state, next_state):
        """Track a decision for hand history"""
        try:
            if session_id not in hand_history_storage:
                hand_history_storage[session_id] = []
            
            # Convert stage enum to integer if needed
            stage = state.get('raw_obs', {}).get('stage', 0)
            if hasattr(stage, 'value'):
                stage = stage.value
            elif not isinstance(stage, int):
                stage = int(stage) if stage else 0
            
            # Get raw_obs safely
            raw_obs = state.get('raw_obs', {})
            
            # IMPORTANT: Get the correct hand for the player
            # For player 0, use stored player_hand from game state (raw_obs contains opponent's hand when it's opponent's turn)
            # For player 1 (opponent), use raw_obs.get('hand') since that's their hand when it's their turn
            hand_cards = []
            if player_id == 0:
                # Get stored player_hand from game state
                if session_id in self.games:
                    stored_player_hand = self.games[session_id].get('player_hand', [])
                    if stored_player_hand:
                        hand_cards = stored_player_hand.copy()
                    else:
                        # Fallback: use raw_obs if stored hand not available (shouldn't happen normally)
                        hand_cards = raw_obs.get('hand', [])
                else:
                    # Fallback: use raw_obs if game state not available
                    hand_cards = raw_obs.get('hand', [])
            else:
                # For opponent (player_id == 1), use raw_obs.get('hand') which contains their hand
                hand_cards = raw_obs.get('hand', [])
            
            # Convert numpy types to native Python types for JSON serialization
            import numpy as np
            from enum import Enum
            
            # Convert action (handle Action enum, numpy types, and primitives)
            if hasattr(action, 'value'):
                # It's an enum (like Action enum from RLCard)
                action = action.value
            elif isinstance(action, (np.integer, np.int64, np.int32)):
                action = int(action)
            elif isinstance(action, (int, float)):
                action = int(action)
            elif isinstance(action, str):
                # Already a string, keep it
                pass
            else:
                # Try to convert to int as fallback
                try:
                    action = int(action)
                except (ValueError, TypeError):
                    action = str(action)
            
            # Convert stage
            if isinstance(stage, (np.integer, np.int64, np.int32)):
                stage = int(stage)
            elif not isinstance(stage, int):
                stage = int(stage) if stage else 0
            
            # Convert pot
            pot = raw_obs.get('pot', 0)
            if isinstance(pot, (np.integer, np.int64, np.int32, np.floating, np.float64)):
                pot = int(pot) if isinstance(pot, (np.integer, np.int64, np.int32)) else int(float(pot))
            else:
                pot = int(pot) if pot else 0
            
            # Convert stakes
            stakes = raw_obs.get('stakes', [0, 0])
            stakes_converted = []
            for s in stakes:
                if isinstance(s, (np.integer, np.int64, np.int32, np.floating, np.float64)):
                    stakes_converted.append(int(s) if isinstance(s, (np.integer, np.int64, np.int32)) else int(float(s)))
                else:
                    stakes_converted.append(int(s) if s else 0)
            
            # Get additional context from game state if available
            all_chips = [0, 0]
            big_blind = 2
            if session_id in self.games:
                game = self.games[session_id]
                # Get chip counts from game state if available
                if hasattr(game.get('env', None), 'game') and hasattr(game['env'].game, 'players'):
                    try:
                        all_chips = [int(p.chips) for p in game['env'].game.players]
                    except:
                        pass
                # Get big blind from stakes if available
                if stakes_converted and len(stakes_converted) >= 2:
                    big_blind = max(stakes_converted) if stakes_converted else 2
            
            # Build decision record
            decision = {
                'player_id': int(player_id) if isinstance(player_id, (np.integer, np.int64, np.int32)) else player_id,
                'action': action,
                'stage': stage,
                'pot': pot,
                'hand': hand_cards,  # Use correct hand based on player_id
                'public_cards': raw_obs.get('public_cards', []),
                'stakes': stakes_converted,
                'all_chips': all_chips,  # Add chip counts for context
                'big_blind': big_blind  # Add big blind for context
            }
            
            hand_history_storage[session_id].append(decision)
            
            # Enhanced logging for tracked decisions
            stage_names = {0: 'Preflop', 1: 'Flop', 2: 'Turn', 3: 'River'}
            stage_name = stage_names.get(stage, f'Stage {stage}')
            player_name = 'Player' if player_id == 0 else 'AI'
            action_name = self._action_to_string(action, None, state, player_id) if hasattr(self, '_action_to_string') else f'Action {action}'

            logger.info(f"📊 Decision tracked - Session: {session_id}, {player_name} (ID: {player_id}), {stage_name}, {action_name} ({action}), Pot: {pot}")
        except Exception as e:
            logger.warning(f"Error tracking decision for session {session_id}: {str(e)}")
            # Don't fail the game if tracking fails

# Initialize game manager
game_manager = GameManager()


@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')


@app.route('/api/game/start', methods=['POST'])
def start_game():
    """Start a new game"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Missing required field: session_id'}), 400
        
        # Start new game
        game_state = game_manager.start_game(session_id)
        
        if game_state is None:
            return jsonify({'error': 'Failed to start game'}), 500
        
        return jsonify(game_state), 200
    
    except ValueError as e:
        logger.warning(f"Invalid input in start_game: {str(e)}")
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error starting game: {error_msg}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred: {error_msg}'}), 500


@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    """Get current game state"""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Missing required parameter: session_id'}), 400
        
        game_state = game_manager.get_game_state(session_id)
        
        if game_state is None:
            return jsonify({'error': 'Game not found'}), 404
        
        return jsonify(game_state), 200
    
    except Exception as e:
        print(f"Error getting game state: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/api/game/action', methods=['POST'])
def process_action():
    """Process a player action"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        session_id = data.get('session_id')
        action_index = data.get('action_index')
        
        if session_id is None:
            return jsonify({'error': 'Missing required field: session_id'}), 400
        
        if action_index is None:
            return jsonify({'error': 'Missing required field: action_index'}), 400
        
        # Check if session exists before processing
        if session_id not in game_manager.games:
            logger.warning(f"Game session not found for action: {session_id}")
            return jsonify({
                'error': 'Game session not found. Please start a new game.',
                'session_id': session_id
            }), 404
        
        # Process action
        game_state = game_manager.process_action(session_id, action_index)
        
        if game_state is None:
            logger.error(f"process_action returned None for session {session_id}, action_index {action_index}")
            return jsonify({
                'error': 'Failed to process action. The game state may be invalid.',
                'session_id': session_id,
                'action_index': action_index
            }), 500
        
        return jsonify(game_state), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error processing action: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/api/game/ai-turn', methods=['POST'])
def process_ai_turn():
    """Process AI opponent's turn"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Missing required field: session_id'}), 400
        
        # Check if session exists before processing
        if session_id not in game_manager.games:
            logger.warning(f"Game session not found for AI turn: {session_id}")
            return jsonify({
                'error': 'Game session not found. Please start a new game.',
                'session_id': session_id
            }), 404
        
        # Process AI turn
        game_state = game_manager.process_ai_turn(session_id)
        
        if game_state is None:
            logger.error(f"process_ai_turn returned None for session {session_id}")
            return jsonify({
                'error': 'Failed to process AI turn. The game state may be invalid.',
                'session_id': session_id
            }), 500
        
        return jsonify(game_state), 200
    
    except ValueError as e:
        logger.warning(f"Invalid input in process_ai_turn: {str(e)}")
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error processing AI turn: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/api/coach/analyze-hand', methods=['POST'])
def analyze_hand():
    """
    Analyze a complete hand with all decisions.
    
    Request:
        {
            "session_id": "string",
            "hand_history": [...],
            "game_state": {...}
        }
    
    Response:
        {
            "overall_grade": "A-F",
            "overall_grade_percentage": 0-100,
            "decisions": [...],
            "key_insights": [...],
            "learning_points": [...]
        }
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'session_id' not in data:
            return jsonify({'error': 'Missing required field: session_id'}), 400
        
        if 'hand_history' not in data:
            return jsonify({'error': 'Missing required field: hand_history'}), 400
        
        if 'game_state' not in data:
            return jsonify({'error': 'Missing required field: game_state'}), 400
        
        session_id = data['session_id']
        hand_history = data['hand_history']
        game_state = data['game_state']
        
        # Validate data types
        if not isinstance(hand_history, list):
            return jsonify({'error': 'hand_history must be a list'}), 400
        
        if not isinstance(game_state, dict):
            return jsonify({'error': 'game_state must be a dictionary'}), 400
        
        # Perform hand analysis with timeout check
        try:
            # Phase 4: Pass session_id and hand_history_storage for pattern recognition
            analysis = evaluator.analyze_hand(
                hand_history, 
                game_state,
                session_id=session_id,
                hand_history_storage=hand_history_storage
            )
            
            # Edge case: Empty hand history (check before analysis)
            if not hand_history or len(hand_history) == 0:
                logger.warning(f"Empty hand history for session {session_id}")
                return jsonify({
                    'overall_grade': 'C',
                    'overall_grade_percentage': 50,
                    'decisions': [],
                    'key_insights': ['No hand history to analyze'],
                    'learning_points': ['Please complete a hand to receive analysis'],
                    'analysis_id': None,
                    'llm_enhanced': False
                }), 200
            
            # Edge case: Invalid game state (check before analysis)
            if not game_state or not isinstance(game_state, dict):
                logger.warning(f"Invalid game state for session {session_id}")
                return jsonify({
                    'error': 'Invalid game state',
                    'overall_grade': 'C',
                    'overall_grade_percentage': 50,
                    'decisions': [],
                    'key_insights': ['Unable to analyze - invalid game state'],
                    'learning_points': ['Please try again'],
                    'analysis_id': None,
                    'llm_enhanced': False
                }), 400
            
            # Check if analysis took too long
            elapsed_time = time.time() - start_time
            if elapsed_time > 2.0:
                # Log warning but still return result
                logger.warning(f"Hand analysis took {elapsed_time:.2f} seconds (target: < 2 seconds)")
            
            return jsonify(analysis), 200
        
        except TimeoutError as e:
            # Analysis timeout
            logger.warning(f"Hand analysis timed out for session {session_id}")
            return jsonify({
                'error': 'Analysis timeout',
                'overall_grade': 'C',
                'overall_grade_percentage': 50,
                'decisions': [],
                'key_insights': ['Analysis took longer than expected'],
                'learning_points': ['Please try again'],
                'analysis_id': None,
                'llm_enhanced': False
            }), 504
        except ValueError as e:
            # Invalid input
            logger.warning(f"Invalid input in hand analysis: {str(e)}")
            return jsonify({
                'error': f'Invalid input: {str(e)}',
                'overall_grade': 'C',
                'overall_grade_percentage': 50,
                'decisions': [],
                'key_insights': ['Unable to analyze - invalid input'],
                'learning_points': ['Please check your request and try again'],
                'analysis_id': None,
                'llm_enhanced': False
            }), 400
        except Exception as e:
            # Calculation error
            logger.error(f"Error during hand analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Analysis error occurred',
                'overall_grade': 'C',
                'overall_grade_percentage': 50,
                'decisions': [],
                'key_insights': ['Unable to complete analysis'],
                'learning_points': ['Please try again'],
                'analysis_id': None,
                'llm_enhanced': False
            }), 500
    
    except Exception as e:
        # Server error
        logger.error(f"Server error in analyze-hand: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'An unexpected error occurred',
            'overall_grade': 'C',
            'overall_grade_percentage': 50,
            'decisions': [],
            'key_insights': ['Error during analysis'],
            'learning_points': ['Please try again'],
            'analysis_id': None,
            'llm_enhanced': False
        }), 500


@app.route('/api/coach/get-hand-history', methods=['GET'])
def get_hand_history():
    """
    Get hand history for a session.
    
    Query params:
        session_id: string (required)
    
    Response:
        {
            "decisions": [...]
        }
    """
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Missing required parameter: session_id'}), 400
        
        # Retrieve hand history from storage
        decisions = hand_history_storage.get(session_id, [])
        
        return jsonify({'decisions': decisions}), 200
    
    except ValueError as e:
        logger.warning(f"Invalid input in get-hand-history: {str(e)}")
        return jsonify({'error': f'Invalid input: {str(e)}', 'decisions': []}), 400
    except Exception as e:
        logger.error(f"Error in get-hand-history: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred', 'decisions': []}), 500


@app.route('/api/coach/chat', methods=['POST'])
def chat():
    """
    Chat with AI coach.
    
    Request:
        {
            "session_id": "string",
            "message": "string",
            "game_context": { ... }  // optional
        }
    
    Response:
        {
            "response": "string",
            "timestamp": "ISO8601"
        }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'session_id' not in data:
            return jsonify({'error': 'Missing required field: session_id'}), 400
        
        if 'message' not in data:
            return jsonify({'error': 'Missing required field: message'}), 400
        
        session_id = data['session_id']
        message = data['message']
        game_context = data.get('game_context')  # Optional
        
        # Validate types
        if not isinstance(session_id, str):
            return jsonify({'error': 'session_id must be a string'}), 400
        
        if not isinstance(message, str):
            return jsonify({'error': 'message must be a string'}), 400
        
        if game_context is not None and not isinstance(game_context, dict):
            return jsonify({'error': 'game_context must be a dictionary'}), 400
        
        # Trim and validate message
        message = message.strip()
        if not message:
            return jsonify({'error': 'message cannot be empty'}), 400
        
        # Call chatbot coach - Phase 3
        if not chatbot_coach:
            return jsonify({
                'error': 'Chatbot coach is unavailable. Please check your API key configuration.',
                'response': 'The AI coach is currently unavailable. Please check your API key configuration.',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }), 503
        
        try:
            # Validate message length
            if len(message) > 500:
                return jsonify({'error': 'Message too long (max 500 characters)'}), 400
            
            # Retrieve hand history for this session
            hand_history = hand_history_storage.get(session_id, [])
            logger.info(f"Chat request for session {session_id}: {len(hand_history)} decisions in hand history")
            if hand_history:
                # Log sample of hand history structure for debugging
                sample = hand_history[-1] if len(hand_history) > 0 else {}
                logger.debug(f"Sample decision structure: {sample}")
                player_decisions = [d for d in hand_history if d.get('player_id') == 0 or d.get('player_id') == "0" or str(d.get('player_id', '')) == "0"]
                logger.info(f"Found {len(player_decisions)} player decisions (player_id == 0) in hand history")
            
            # Call chatbot coach with timeout handling
            response = chatbot_coach.chat(
                message=message,
                game_context=game_context,
                hand_history=hand_history,
                session_id=session_id
            )
            
            return jsonify(response), 200
        
        except TimeoutError:
            logger.warning(f"Chat request timed out for session {session_id}")
            # Try rule-based fallback before returning error
            if chatbot_coach:
                try:
                    rule_based_response = chatbot_coach._generate_rule_based_response(
                        message, game_context, hand_history
                    )
                    if rule_based_response:
                        logger.info("Using rule-based response after timeout")
                        return jsonify(rule_based_response), 200
                except Exception as fallback_error:
                    logger.warning(f"Rule-based fallback failed: {fallback_error}")
            
            return jsonify({
                'error': 'Request timed out',
                'response': 'I\'m having trouble connecting right now. Please try again in a moment.',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }), 504
        except Exception as e:
            # API failure - try rule-based fallback first
            logger.error(f"Error in chatbot coach: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try rule-based fallback before returning error
            if chatbot_coach:
                try:
                    rule_based_response = chatbot_coach._generate_rule_based_response(
                        message, game_context, hand_history
                    )
                    if rule_based_response:
                        logger.info("Using rule-based response after exception")
                        return jsonify(rule_based_response), 200
                except Exception as fallback_error:
                    logger.warning(f"Rule-based fallback failed: {fallback_error}")
            
            # Return fallback response only if rule-based also failed
            return jsonify({
                'error': 'Chat service temporarily unavailable',
                'response': 'I\'m having trouble processing your question right now. Please try again in a moment.',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }), 500
    
    except Exception as e:
        # Server error
        logger.error(f"Server error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'An unexpected error occurred',
            'response': 'I apologize, but I encountered an error. Please try again.',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500


@app.route('/api/coach/analyze-hand-async/<analysis_id>', methods=['GET'])
def get_async_analysis(analysis_id):
    """
    Get async LLM-enhanced analysis results.
    
    Query params:
        analysis_id: Analysis ID from initial analyze-hand response
    
    Response:
        {
            "ready": bool,
            "llm_insights": list,
            "enhanced_explanations": dict,
            "pattern_insights": list,
            "llm_learning_points": list
        }
    """
    try:
        if not analysis_id:
            return jsonify({'error': 'Missing required parameter: analysis_id'}), 400
        
        # Get async results from evaluator
        async_results = evaluator.get_async_llm_results(analysis_id)
        
        if async_results is None:
            # Not ready yet
            return jsonify({
                'ready': False,
                'llm_insights': [],
                'enhanced_explanations': {},
                'pattern_insights': [],
                'llm_learning_points': [],
                'api_unavailable': False
            }), 200
        
        # Return enhanced content
        return jsonify({
            'ready': async_results.get('ready', False),
            'llm_insights': async_results.get('llm_insights', []),
            'enhanced_explanations': async_results.get('enhanced_explanations', {}),
            'pattern_insights': async_results.get('pattern_insights', []),
            'llm_learning_points': async_results.get('llm_learning_points', []),
            'api_unavailable': async_results.get('api_unavailable', False),
            'error': async_results.get('error')
        }), 200
    
    except ValueError as e:
        logger.warning(f"Invalid input in get_async_analysis: {str(e)}")
        return jsonify({
            'error': f'Invalid input: {str(e)}',
            'ready': False,
            'llm_insights': [],
            'enhanced_explanations': {},
            'pattern_insights': [],
            'llm_learning_points': [],
            'api_unavailable': False
        }), 400
    except Exception as e:
        logger.error(f"Error in get_async_analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'An unexpected error occurred',
            'ready': False,
            'llm_insights': [],
            'enhanced_explanations': {},
            'pattern_insights': [],
            'llm_learning_points': [],
            'api_unavailable': False
        }), 500


@app.route('/api/admin/remove-session/<session_id>', methods=['DELETE'])
def remove_session(session_id):
    """
    Remove a session from logging/memory.

    Args:
        session_id: The session ID to remove

    Returns:
        JSON response indicating success/failure
    """
    try:
        removed_from_history = False
        removed_from_games = False
        history_count = 0

        # Remove from hand history storage
        if session_id in hand_history_storage:
            history_count = len(hand_history_storage[session_id])
            del hand_history_storage[session_id]
            removed_from_history = True
            logger.info(f"🗑️ Removed session {session_id} from hand_history_storage ({history_count} decisions)")

        # Remove from game manager
        if session_id in game_manager.games:
            del game_manager.games[session_id]
            removed_from_games = True
            logger.info(f"🗑️ Removed session {session_id} from game_manager.games")

        if removed_from_history or removed_from_games:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'removed_from_history': removed_from_history,
                'removed_from_games': removed_from_games,
                'decisions_removed': history_count
            }), 200
        else:
            return jsonify({
                'success': False,
                'session_id': session_id,
                'message': 'Session not found'
            }), 404

    except Exception as e:
        logger.error(f"Error removing session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'session_id': session_id,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
