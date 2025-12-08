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
try:
    import rlcard
    from rlcard.agents import RandomAgent
    from rlcard.games.nolimitholdem.round import Action
except ImportError:
    # Use mock implementation when rlcard is not available
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import rlcard_mock as rlcard
    from rlcard_mock.agents import RandomAgent
    from rlcard_mock import Action

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Environment variables should be set manually or through other means")

from coach.strategy_evaluator import StrategyEvaluator
from coach.action_labeling import ActionLabeling
from coach.action_validator import ActionValidator
from coach.chatbot_coach import ChatbotCoach
from coach.gto_agent import GTOAgent
from coach.llm_opponent_agent import LLMOpponentAgent
from coach.pot_calculator import calculate_pot_from_state, pot_to_bb, calculate_pot
from coach.hand_events import HandEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Disable werkzeug HTTP request logging (suppress lines like "GET /api/game/state HTTP/1.1" 200)
logging.getLogger('werkzeug').setLevel(logging.WARNING)


def convert_card_ints(obj):
    """Convert card integers and RLCard Card objects to string format for frontend display"""
    if obj is None:
        return obj
    
    # Handle list of cards
    if isinstance(obj, list):
        result = []
        for card in obj:
            try:
                # Try to get Card object index first
                from rlcard.games.base import Card
                if isinstance(card, Card):
                    card_index = card.get_index()
                else:
                    # Convert to int (handles numpy types)
                    card_index = int(card)
            except (ImportError, ValueError, TypeError, AttributeError):
                # If conversion fails, try direct int conversion
                try:
                    card_index = int(card)
                except (ValueError, TypeError):
                    # If it's already a string, keep it
                    result.append(str(card))
                    continue
            
            # Convert card index (0-51) to string format (e.g., "SA", "KH")
            if 0 <= card_index <= 51:
                suits = ['S', 'H', 'D', 'C']  # Spades, Hearts, Diamonds, Clubs
                ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
                suit_idx = card_index // 13
                rank_idx = card_index % 13
                suit = suits[suit_idx]
                rank = ranks[rank_idx]
                result.append(suit + rank)
            else:
                result.append(str(card))
        return result
    
    # Handle single Card object
    try:
        from rlcard.games.base import Card
        if isinstance(obj, Card):
            card_index = obj.get_index()
            suits = ['S', 'H', 'D', 'C']
            ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
            suit_idx = card_index // 13
            rank_idx = card_index % 13
            suit = suits[suit_idx]
            rank = ranks[rank_idx]
            return suit + rank
    except (ImportError, AttributeError):
        pass
    
    # If it's already a string or something else, return as-is
    return obj

def convert_numpy_types(obj):
    """Convert numpy types and RLCard objects to native Python types for JSON serialization"""
    # Handle RLCard Card objects first
    try:
        from rlcard.games.base import Card
        if isinstance(obj, Card):
            return obj.get_index()
    except ImportError:
        pass

    # Handle Action enum objects
    try:
        import rlcard
        if hasattr(obj, 'value') and hasattr(obj, '__class__'):
            # Check if it's an Action enum
            if obj.__class__.__name__ == 'Action':
                return obj.value
    except ImportError:
        pass

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

def safe_int_convert(value, default=0):
    """Safely convert a value to int, handling NumPy arrays and None values.
    
    This function prevents "The truth value of an array with more than one element is ambiguous" errors
    by properly handling NumPy arrays before boolean checks.
    
    Args:
        value: The value to convert (can be int, NumPy array, None, etc.)
        default: Default value to return if conversion fails or value is None/empty
    
    Returns:
        int: The converted integer value
    """
    if value is None:
        return default
    
    # Handle NumPy arrays first (before any boolean checks)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        # Extract scalar value from array
        try:
            return int(value.item())
        except (ValueError, OverflowError):
            # If item() fails, try indexing
            return int(value[0]) if value.size > 0 else default
    
    # Handle NumPy scalar types
    if isinstance(value, (np.integer, np.floating)):
        return int(value)
    
    # Handle objects with .value attribute (enums, etc.)
    if hasattr(value, 'value'):
        return safe_int_convert(value.value, default)
    
    # For regular Python types, check if truthy before converting
    # This is safe now because we've already handled arrays
    try:
        if value:
            return int(value)
        else:
            return default
    except (ValueError, TypeError):
        return default


def safe_bool_convert(value, default=False):
    """Safely convert a value to bool, handling NumPy arrays and None values.
    
    This function prevents "The truth value of an array with more than one element is ambiguous" errors
    by properly handling NumPy arrays before boolean checks.
    
    Args:
        value: The value to convert (can be bool, NumPy array, None, etc.)
        default: Default value to return if conversion fails or value is None/empty
    
    Returns:
        bool: The converted boolean value
    """
    if value is None:
        return default
    
    # Handle NumPy arrays first (before any boolean checks)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        # Extract scalar value from array and convert to bool
        try:
            return bool(value.item())
        except (ValueError, OverflowError):
            # If item() fails, try indexing
            return bool(value[0]) if value.size > 0 else default
    
    # Handle NumPy scalar types
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return bool(value)
    
    # Handle objects with .value attribute (enums, etc.)
    if hasattr(value, 'value'):
        return safe_bool_convert(value.value, default)
    
    # For regular Python types, convert to bool
    try:
        return bool(value)
    except (ValueError, TypeError):
        return default


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

# Disable Flask's automatic .env loading
import os
os.environ['FLASK_SKIP_DOTENV'] = '1'

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
        logger.info(f"=== GAME_MANAGER.START_GAME({session_id}) CALLED ===")

        # Create environment with BB-first action order support
        # Use a custom environment that handles the game creation properly
        try:
            logger.info("Attempting to create RLCard environment...")
            env = rlcard.make('no-limit-holdem')
            logger.info("RLCard environment created successfully")
        except Exception as e:
            logger.warning(f"Exception creating RLCard environment: {type(e).__name__}: {e}")
            if "'NolimitholdemGame' object has no attribute 'configure'" in str(e):
                # Fallback: Use mock environment if RLCard patching failed
                logger.info("Falling back to mock environment...")
                from rlcard_mock import MockEnvironment
                env = MockEnvironment()
                logger.info("Using mock environment due to RLCard patching failure")
            else:
                logger.error(f"Raising exception: {e}")
                raise e

        logger.debug(f"Environment created: {type(env)}")
        logger.debug(f"Environment has game attribute: {hasattr(env, 'game')}")
        
        # Detailed logging for environment structure debugging
        if hasattr(env, 'game'):
            logger.info(f"Environment has game object: {type(env.game)}")
            logger.info(f"Game has init_game: {hasattr(env.game, 'init_game')}")
            logger.info(f"Game has step: {hasattr(env.game, 'step')}")
            logger.info(f"Game has __class__: {hasattr(env.game, '__class__')}")
        else:
            logger.warning("Environment does NOT have game attribute")

        # PATCH: Apply BB-first action order logic to the game
        # Only patch if the game has the required methods (real RLCard games)
        if (hasattr(env, 'game') and hasattr(env.game, '__class__') and
            hasattr(env.game, 'init_game') and hasattr(env.game, 'step')):
            logger.info("Applying BB-first patches to game...")
            original_init_game = env.game.init_game
            original_step = env.game.step

            # Create wrapper functions that properly handle self parameter
            def patched_init_game_wrapper(self):
                # FIX: Ensure dealer_id is randomly selected using secrets module
                # This fixes the issue where deployment always defaults to the same dealer
                # because random module might not be properly seeded in deployment environments
                # secrets module uses OS-provided entropy and doesn't require seeding
                import secrets
                
                # Set dealer_id to a random value BEFORE calling original_init_game
                # The original code checks: if dealer_id is None, use np_random.choice, else use modulo
                # By setting it to a non-None value, we bypass the np_random.choice call
                random_dealer_id = secrets.choice(range(self.num_players))
                self.dealer_id = random_dealer_id
                logger.info(f"🎲 [DEALER_FIX] Set dealer_id to {self.dealer_id} using secrets module (before init_game)")
                
                # Call original init_game (it will use our dealer_id and apply modulo if needed)
                result = original_init_game()
                state, current_player = result
                
                # Log the final dealer_id after init_game
                logger.info(f"🎲 [DEALER_FIX] Final dealer_id after init_game: {self.dealer_id}")

                # Apply BB-first logic for 2-player games
                if self.num_players == 2:
                    # SB (player 1) acts first preflop (already set by original logic)
                    # For postflop, we'll handle in step()
                    pass

                return result

            def patched_step_wrapper(self, action):
                # Call original step
                result = original_step(action)
                next_state, next_player_id = result

                # Apply BB-first logic for postflop stages in 2-player games
                if (self.num_players == 2 and
                    hasattr(self, 'round_counter') and self.round_counter >= 1 and
                    hasattr(self, 'dealer_id')):
                    # Postflop: BB (dealer) acts first
                    self.game_pointer = self.dealer_id

                return result

            # Monkey patch the methods - bind properly to preserve self
            import types
            env.game.init_game = types.MethodType(patched_init_game_wrapper, env.game)
            env.game.step = types.MethodType(patched_step_wrapper, env.game)
            logger.info("BB-first patches applied successfully")
        else:
            logger.info("Skipping BB-first patches - not a standard RLCard environment")

        logger.debug(f"Environment num_actions: {env.num_actions}")

        # Create agents
        logger.info("Creating human agent...")
        human_agent = WebHumanAgent(env.num_actions)
        logger.info("Creating AI agent...")
        # Use LLM opponent agent instead of GTO agent for LLM-powered decisions
        # Falls back to GTOAgent on LLM failures
        ai_agent = LLMOpponentAgent(num_actions=env.num_actions)
        logger.info("Setting agents on environment...")
        env.set_agents([human_agent, ai_agent])

        # Reset environment
        logger.info("Resetting environment...")
        state, player_id = env.reset()
        logger.info(f"Environment reset successful. Player ID: {player_id}")
        logger.debug(f"Initial state keys: {list(state.keys()) if isinstance(state, dict) else type(state)}")
        
        # Store game state
        # Get human player's hand (player 0) from the game
        # In RLCard, hands are dealt at the start, so both players have hands
        player_hand = []
        try:
            # Try to get human player's hand directly from game object
            if hasattr(env, 'game') and hasattr(env.game, 'players') and len(env.game.players) > 0:
                human_player = env.game.players[0]  # Player 0 is always human
                if hasattr(human_player, 'hand'):
                    player_hand = list(human_player.hand) if human_player.hand else []
                    # Log the actual cards being dealt for debugging
                    logger.info(f"🎴 [CARD_DEAL] Player 0 hand from game.players[0]: {player_hand}")
                    # Convert to readable format for logging
                    if player_hand:
                        try:
                            card_strs = [convert_card_ints([c])[0] if isinstance(c, (int, type(None))) else str(c) for c in player_hand]
                            logger.info(f"🎴 [CARD_DEAL] Player 0 hand (readable): {card_strs}")
                        except:
                            logger.info(f"🎴 [CARD_DEAL] Player 0 hand (raw): {player_hand}")
            else:
                # Fallback: if player 0's turn, get from raw_obs
                initial_raw_obs = state.get('raw_obs', {})
                if player_id == 0:
                    player_hand = initial_raw_obs.get('hand', [])
                    logger.info(f"🎴 [CARD_DEAL] Player 0 hand from raw_obs (player_id=0): {player_hand}")
                else:
                    # If AI goes first, we need to get player 0's hand from the game state
                    # In RLCard, we can get state from player 0's perspective
                    player_0_state = env.get_state(0)
                    player_hand = player_0_state.get('raw_obs', {}).get('hand', [])
                    logger.info(f"🎴 [CARD_DEAL] Player 0 hand from get_state(0) (player_id={player_id}): {player_hand}")
        except Exception as e:
            # Final fallback
            logger.warning(f"🎴 [CARD_DEAL] Exception getting player hand: {e}")
            initial_raw_obs = state.get('raw_obs', {})
            if player_id == 0:
                player_hand = initial_raw_obs.get('hand', [])
                logger.info(f"🎴 [CARD_DEAL] Player 0 hand from final fallback raw_obs: {player_hand}")
        
        # Log final stored hand
        logger.info(f"🎴 [CARD_DEAL] Final stored player_hand for session {session_id}: {player_hand}")
        
        self.games[session_id] = {
            'env': env,
            'human_agent': human_agent,
            'ai_agent': ai_agent,
            'current_state': state,
            'current_player': player_id,
            'hand_history': [],
            'last_stage': 0,  # Track previous stage for detecting community card deals
            'last_public_cards_count': 0,  # Track previous public cards count
            'action_history_with_cards': [],  # Store action history with community cards (legacy)
            'last_action_count': 0,  # Track number of actions processed
            'blinds_added': False,  # Track if blind entries have been added
            'player_hand': player_hand.copy() if player_hand else [],  # Store player's hand for the entire hand
            'hand_events': []  # New event-based history (single source of truth)
        }
        
        # Emit blind events immediately after game initialization
        logger.debug("Emitting blind events...")
        self._emit_blind_events(session_id, env)

        logger.info("Getting final game state...")
        final_state = self.get_game_state(session_id)
        logger.debug(f"Final game state type: {type(final_state)}")
        logger.info("=== GAME_MANAGER.START_GAME() COMPLETED SUCCESSFULLY ===")
        return final_state
    
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
        
        # Track game state for context
        last_stage = game.get('last_stage', -1)
        current_stage = raw_obs.get('stage', 0)
        if hasattr(current_stage, 'value'):
            current_stage = current_stage.value
        elif not isinstance(current_stage, int):
            current_stage = safe_int_convert(current_stage, 0)

        # Detect community card deals and emit events
        last_public_cards_count = game.get('last_public_cards_count', 0)
        current_public_cards = raw_obs.get('public_cards', [])
        current_public_cards_count = len(current_public_cards)
        
        # Check if new community cards were dealt
        if current_public_cards_count > last_public_cards_count:
            stage_names = {1: 'flop', 2: 'turn', 3: 'river'}
            stage_name = stage_names.get(current_stage, '')
            
            if stage_name == 'flop' and current_public_cards_count >= 3:
                # Flop dealt
                new_cards = current_public_cards[:3]
                pot = raw_obs.get('pot', 0)
                self._emit_community_card_event(session_id, 'Flop', new_cards, pot)
            elif stage_name == 'turn' and current_public_cards_count >= 4:
                # Turn dealt
                new_cards = [current_public_cards[3]]
                pot = raw_obs.get('pot', 0)
                self._emit_community_card_event(session_id, 'Turn', new_cards, pot)
            # River card events are not added to action history UI
            # elif stage_name == 'river' and current_public_cards_count >= 5:
            #     # River dealt
            #     new_cards = [current_public_cards[4]]
            #     pot = raw_obs.get('pot', 0)
            #     self._emit_community_card_event(session_id, 'River', new_cards, pot)
        
        game['last_public_cards_count'] = current_public_cards_count
        game['last_stage'] = current_stage

        # Track if game is over for next iteration
        was_over = game.get('was_over', False)
        is_over = safe_bool_convert(env.is_over())
        game['was_over'] = is_over
        
        # Detect hand end and emit win event
        if is_over and not was_over:
            # Hand just ended, emit win event
            payoffs = env.get_payoffs() if hasattr(env, 'get_payoffs') else None
            if payoffs is not None and len(payoffs) > 0:
                # Find winner (player with positive payoff)
                for player_id, payoff in enumerate(payoffs):
                    # Convert payoff to native Python type to avoid numpy comparison issues
                    payoff_value = float(payoff) if hasattr(payoff, '__float__') else payoff
                    if payoff_value > 0:
                        big_blind = raw_obs.get('big_blind', 2)
                        pot = raw_obs.get('pot', 0)
                        # Convert to int to avoid array comparison errors
                        big_blind = safe_int_convert(big_blind, 2)
                        pot = safe_int_convert(pot, 0)
                        pot_bb = pot / big_blind if big_blind > 0 else 0
                        self._emit_win_event(session_id, player_id, pot_bb)
                        break
        
        # Get human player's hand (player 0) directly from the game object
        # This ensures we always have the correct hand regardless of whose turn it is
        # CRITICAL: Always return player 0's hand, never the current player's hand
        player_hand = []
        try:
            if hasattr(env, 'game') and hasattr(env.game, 'players') and len(env.game.players) > 0:
                human_player = env.game.players[0]  # Player 0 is always the human
                if hasattr(human_player, 'hand'):
                    player_hand = list(human_player.hand) if human_player.hand else []
            else:
                # Fallback: Get state from player 0's perspective if direct access fails
                try:
                    player_0_state = env.get_state(0)
                    player_hand = player_0_state.get('raw_obs', {}).get('hand', [])
                except Exception:
                    # If get_state fails, continue to next fallback
                    pass
        except Exception as e:
            # If direct access fails, try to get player 0's state
            try:
                player_0_state = env.get_state(0)
                player_hand = player_0_state.get('raw_obs', {}).get('hand', [])
            except Exception:
                # If get_state also fails, continue to stored hand fallback
                pass
        
        # Final fallback: Use stored hand from game initialization
        # This is the most reliable source since it was captured at game start
        if not player_hand:
            player_hand = game.get('player_hand', [])
            if player_hand:
                logger.debug(f"🎯 [GET_GAME_STATE] Using stored player hand as fallback: {player_hand}")
        
        # Last resort: Only use raw_obs if it's player 0's turn (to avoid getting wrong player's hand)
        if not player_hand and current_player == 0:
            player_hand = raw_obs.get('hand', [])
            if player_hand:
                logger.debug(f"🎯 [GET_GAME_STATE] Using raw_obs hand (player 0's turn): {player_hand}")
        
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
        
        # Get legal actions - try real RLCard location first (top level of state), then mock location (raw_obs)
        legal_actions = convert_actions(state.get('legal_actions', raw_obs.get('legal_actions', [])))
        raw_legal_actions = convert_actions(state.get('raw_legal_actions', raw_obs.get('raw_legal_actions', [])))

        # Fallback: If no legal actions in state, try to get them from environment
        if not raw_legal_actions:
            if hasattr(env, 'get_legal_actions'):
                try:
                    env_legal_actions = env.get_legal_actions()
                    raw_legal_actions = convert_actions(env_legal_actions)
                    legal_actions = raw_legal_actions.copy()
                    logger.debug(f"Got legal actions from env.get_legal_actions(): {raw_legal_actions}")
                except Exception as e:
                    logger.warning(f"Failed to get legal actions from environment: {e}")

            # Last resort: Provide basic poker actions if still no legal actions
            if not raw_legal_actions:
                logger.warning(f"No legal actions found in state or environment, using default poker actions for session {session_id}")
                # Basic poker actions: Fold, Check/Call, Raise Half Pot, Raise Pot, All-in
                raw_legal_actions = [0, 1, 2, 3, 4]
                legal_actions = [0, 1, 2, 3, 4]
            else:
                # Ensure basic actions are always available as fallback
                basic_actions = [0, 1, 4]  # Fold, Check/Call, All-in should always be available
                for action in basic_actions:
                    if action not in legal_actions:
                        legal_actions.append(action)
                        logger.debug(f"Added basic action {action} to legal_actions")
                    if action not in raw_legal_actions:
                        raw_legal_actions.append(action)
                        logger.debug(f"Added basic action {action} to raw_legal_actions")

        # Convert stage enum to integer if needed
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        elif not isinstance(stage, int):
            stage = safe_int_convert(stage, 0)

        # Get dealer_id from game object (not in raw_obs)
        dealer_id = None
        if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
            dealer_id = env.game.dealer_id
            # Convert to int to avoid NumPy array comparison errors
            dealer_id = safe_int_convert(dealer_id, None)

        # Calculate button_id (small blind in heads-up)
        button_id = None
        if dealer_id is not None:
            button_id = (dealer_id + 1) % 2  # In HUNL, button = (dealer + 1) % 2
            # Convert to int to avoid NumPy array comparison errors
            button_id = safe_int_convert(button_id, None)

        # Convert current_player to int to avoid NumPy array comparison errors
        current_player = safe_int_convert(current_player, 0)

        # Get in_chips from players (amount each player has bet in current hand)
        in_chips = [0, 0]
        # Also try to get remaining chips directly from player objects (more reliable than raw_obs)
        remaining_chips = [0, 0]
        if hasattr(env, 'game') and hasattr(env.game, 'players'):
            try:
                in_chips = [int(p.in_chips) for p in env.game.players]
                # Try to get remaining chips from player objects if available
                # RLCard players might have a 'remained_chips' or 'chips' attribute
                for i, player in enumerate(env.game.players):
                    if hasattr(player, 'remained_chips'):
                        remaining_chips[i] = int(player.remained_chips)
                    elif hasattr(player, 'chips'):
                        remaining_chips[i] = int(player.chips)
                    elif hasattr(player, 'in_chips') and hasattr(env.game, 'init_chips'):
                        # Calculate: starting chips - chips bet
                        starting = env.game.init_chips[i] if hasattr(env.game, 'init_chips') and i < len(env.game.init_chips) else 100
                        remaining_chips[i] = starting - int(player.in_chips)
                logger.debug(f"💰 [STACKS] Remaining chips from players: {remaining_chips}")
            except Exception as e:
                logger.debug(f"💰 [STACKS] Error getting chips from players: {e}")
                pass
        
        # Get raised array (amount bet in current betting round)
        # SIMPLIFIED: Use RLCard's natural raised array directly
        # With native BB-first action order, RLCard's state should be consistent
        raised = raw_obs.get('raised', [0, 0])
        
        # Fix for small blind first to act preflop - ensure standard raise options are available
        # Use ActionLabeling context to determine when "Raise to 3BB" should be available
        if (stage == 0 and  # Preflop
            current_player == 0 and  # Human player
            not safe_bool_convert(env.is_over())):
            
            # Build context to check if "Raise to 3BB" should be available
            state_with_processed = {
                'raw_obs': raw_obs,
                'in_chips': in_chips,
                'raised': raised
            }
            
            try:
                from coach.action_labeling import ActionLabeling
                context = ActionLabeling.get_context_from_state(state_with_processed, player_id=0, env=env)
                button_labels = ActionLabeling.get_button_labels(context)
                
                logger.debug(f"🔧 [GET_GAME_STATE] Preflop SB check - context: {context}, button_labels: {button_labels}, button_id: {button_id}")
                
                # Check if "Raise to 3BB" should be available (check both label formats)
                raise_label = button_labels.get('raiseHalfPot', '')
                should_show = button_labels.get('showRaiseHalfPot', False)
                is_small_blind = context.get('is_small_blind', False)
                is_first_to_act = context.get('is_first_to_act', False)
                
                # If we're SB first to act preflop and should show "Raise to 3BB", add action 2
                # Check for both "Raise to 3 BB" (with space) and "Raise to 3BB" (without space)
                should_add_action_2 = (
                    should_show and 
                    ('Raise to 3' in raise_label or raise_label == 'Raise to 3BB') and
                    (is_small_blind and is_first_to_act)
                )
                
                # Also check if button_id indicates SB (more reliable than context detection)
                if button_id == 0 and is_first_to_act:
                    should_add_action_2 = True
                
                if should_add_action_2:
                    if 2 not in legal_actions:
                        legal_actions.append(2)
                        logger.info(f"🔧 [GET_GAME_STATE] Added action 2 (Raise to 3BB) to legal_actions - label: '{raise_label}', show: {should_show}, SB: {is_small_blind}, first: {is_first_to_act}, button_id: {button_id}")
                    if 2 not in raw_legal_actions:
                        raw_legal_actions.append(2)
                        logger.info(f"🔧 [GET_GAME_STATE] Added action 2 (Raise to 3BB) to raw_legal_actions")
                else:
                    logger.debug(f"🔧 [GET_GAME_STATE] Not adding action 2 - label: '{raise_label}', show: {should_show}, SB: {is_small_blind}, first: {is_first_to_act}, button_id: {button_id}")
            except Exception as e:
                logger.warning(f"Failed to use ActionLabeling context for action 2 fix: {e}")
                logger.exception(e)
            
            # Aggressive fallback: If preflop and user's turn, always add action 2 if not present
            # This ensures "Raise to 3BB" is always available for SB first to act
            if 2 not in legal_actions:
                legal_actions.append(2)
                logger.info(f"🔧 [GET_GAME_STATE] Aggressive fallback: Added action 2 to legal_actions (preflop, user turn, button_id={button_id})")
            if 2 not in raw_legal_actions:
                raw_legal_actions.append(2)
                logger.info(f"🔧 [GET_GAME_STATE] Aggressive fallback: Added action 2 to raw_legal_actions (preflop, user turn, button_id={button_id})")
        
        # Calculate pot accurately using centralized pot calculator with reconstructed raised array
        # Create a modified raw_obs with the reconstructed raised array
        modified_raw_obs = raw_obs.copy()
        modified_raw_obs['raised'] = raised
        # Also include in_chips in raw_obs if available (for postflop cumulative pot calculation)
        if in_chips and len(in_chips) >= 2:
            modified_raw_obs['in_chips'] = in_chips
        pot = calculate_pot_from_state(modified_raw_obs, env)
        pot_bb = pot_to_bb(pot, raw_obs.get('big_blind', 2))

        # Calculate remaining stacks (stakes)
        # Priority order:
        # 1. Remaining chips from player objects (most reliable)
        # 2. stakes from raw_obs (if available and valid)
        # 3. Calculate from all_chips - in_chips
        # 4. Use all_chips directly
        # 5. Default to 50 BB stacks (100 chips) - RLCard default
        
        all_chips = raw_obs.get('all_chips', None)
        raw_stakes = raw_obs.get('stakes', None)
        
        # Log what RLCard is actually providing for debugging
        logger.debug(f"💰 [STACKS_DEBUG] raw_obs keys: {list(raw_obs.keys())}")
        logger.debug(f"💰 [STACKS_DEBUG] stakes from raw_obs: {raw_stakes}")
        logger.debug(f"💰 [STACKS_DEBUG] all_chips from raw_obs: {all_chips}")
        logger.debug(f"💰 [STACKS_DEBUG] in_chips from players: {in_chips}")
        logger.debug(f"💰 [STACKS_DEBUG] remaining_chips from players: {remaining_chips}")
        
        # Priority 1: Use remaining chips from player objects if available and valid
        if remaining_chips and len(remaining_chips) >= 2 and (remaining_chips[0] > 0 or remaining_chips[1] > 0):
            stakes = [int(remaining_chips[0]), int(remaining_chips[1])]
        # Priority 2: Use stakes from raw_obs if available and valid
        elif raw_stakes and isinstance(raw_stakes, list) and len(raw_stakes) >= 2 and not (raw_stakes[0] == 0 and raw_stakes[1] == 0):
            stakes = [int(raw_stakes[0]), int(raw_stakes[1])]
            logger.info(f"💰 [STACKS] Using stakes from raw_obs: {stakes}")
        # Priority 3: Calculate from all_chips - in_chips
        elif all_chips and len(all_chips) >= 2 and in_chips and len(in_chips) >= 2:
            stakes = [int(all_chips[0]) - int(in_chips[0]), int(all_chips[1]) - int(in_chips[1])]
            logger.info(f"💰 [STACKS] Calculated stakes from all_chips - in_chips: {stakes}")
        # Priority 4: Use all_chips directly if in_chips not available
        elif all_chips and len(all_chips) >= 2:
            stakes = [int(all_chips[0]), int(all_chips[1])]
            logger.info(f"💰 [STACKS] Using all_chips as stakes: {stakes}")
        # Priority 5: Final fallback - default to 50 BB stacks (100 chips each)
        else:
            big_blind = raw_obs.get('big_blind', 2)
            default_chips = 100  # 50 BB (RLCard default)
            stakes = [default_chips, default_chips]
            logger.warning(f"💰 [STACKS] FALLBACK: Using default 50 BB stacks: {stakes} (raw_stakes={raw_stakes}, all_chips={all_chips}, in_chips={in_chips}, remaining_chips={remaining_chips})")
        
        # Ensure stakes is a list of 2 integers
        if not isinstance(stakes, list) or len(stakes) < 2:
            big_blind = raw_obs.get('big_blind', 2)
            default_chips = 100  # 50 BB (RLCard default)
            stakes = [default_chips, default_chips]
            logger.warning(f"💰 [STACKS] Invalid stakes format, using default: {stakes}")

        # Build game state response
        game_state = {
            'hand': player_hand,  # Use stored player hand, not raw_obs hand
            'public_cards': raw_obs.get('public_cards', []),
            'stakes': stakes,  # Use calculated stakes instead of raw_obs.get('stakes', [0, 0])
            'pot': pot,  # Use calculated pot instead of raw_obs pot
            'stage': stage,
            'legal_actions': legal_actions,
            'raw_legal_actions': raw_legal_actions,
            'current_player': current_player,
            'is_waiting_for_action': not safe_bool_convert(env.is_over()) and current_player == 0,
            'is_over': safe_bool_convert(env.is_over()),
            'big_blind': raw_obs.get('big_blind', 2),
            'button_id': button_id,
            'dealer_id': dealer_id,
            'raised': raised,
            'in_chips': in_chips,  # Total chips each player has put in this hand
            'all_chips': raw_obs.get('all_chips', [100, 100]),  # Starting chips (50 BB each, for debugging)
            'action_history': self._build_action_history(env, state, session_id),  # Legacy format
            'hand_events': [event.to_dict() for event in game.get('hand_events', [])],  # New event-based format
            'payoffs': env.get_payoffs() if safe_bool_convert(env.is_over()) else None,
            'opponent_hand': self._get_opponent_hand(env) if safe_bool_convert(env.is_over()) else None
        }
        
        # Convert numpy types to native Python types for JSON serialization first
        game_state = convert_numpy_types(game_state)
        
        # Convert card integers to strings for frontend display (after numpy conversion)
        # CRITICAL: Always convert cards to string format for frontend
        # Force conversion by explicitly checking and converting
        hand = game_state.get('hand', [])
        if hand and isinstance(hand, list) and len(hand) > 0:
            converted = convert_card_ints(hand)
            if converted and isinstance(converted, list):
                game_state['hand'] = converted
        
        public_cards = game_state.get('public_cards', [])
        if public_cards and isinstance(public_cards, list):
            converted = convert_card_ints(public_cards)
            if converted and isinstance(converted, list):
                game_state['public_cards'] = converted
        
        opponent_hand = game_state.get('opponent_hand')
        if opponent_hand and isinstance(opponent_hand, list) and len(opponent_hand) > 0:
            converted = convert_card_ints(opponent_hand)
            if converted and isinstance(converted, list):
                game_state['opponent_hand'] = converted

        # Final pot recalculation using reconstructed raised array (already done above, but ensure consistency)
        # The pot was already calculated with reconstructed raised array above, so this is just for logging
        final_pot = calculate_pot(raised, game_state.get('big_blind', 2), dealer_id, stage)
        # Convert to int to avoid array comparison errors
        final_pot = safe_int_convert(final_pot, 0)
        pot = safe_int_convert(pot, 0)
        if final_pot > 0 and final_pot != pot:
            game_state['pot'] = final_pot
        else:
            game_state['pot'] = pot

        return game_state
    
    def _build_action_history(self, env, state, session_id):
        """Build action history - SIMPLIFIED for native RLCard integration"""
        game = self.games.get(session_id, {})

        # Initialize history if not exists
        if 'action_history_with_cards' not in game:
            game['action_history_with_cards'] = []

        try:
            # SIMPLIFIED: With native RLCard, use the hand_history_storage directly
            # The _track_decision method now maintains accurate history
            history = []  # Initialize history to avoid scoping issues

            # Add blind entries at the beginning if they were added before
            if game.get('blinds_added', False):
                self._add_blind_entries(history, env, session_id)

            if session_id in hand_history_storage:
                # Convert decision records to action history format
                for decision in hand_history_storage[session_id]:
                    bet_amount = decision.get('bet_amount', None)
                    stored_context = decision.get('context')

                    # Use stored context if available, otherwise fall back to current state reconstruction
                    if stored_context is not None:
                        # Use the stored context for accurate action labeling
                        action_label = ActionLabeling.get_action_label(decision['action'], stored_context, bet_amount=bet_amount)
                    else:
                        # Fallback to original method if no stored context
                        action_label = self._action_to_string(decision['action'], env, state, decision['player_id'], bet_amount)

                    action_entry = {
                        'type': 'action',
                        'player_id': decision['player_id'],
                        'player_name': 'You' if decision['player_id'] == 0 else 'Opponent',
                        'action': action_label,
                        'stage': decision['stage'],
                        'pot': decision['pot'],
                        'hand': decision.get('hand', []),
                        'public_cards': decision.get('public_cards', [])
                    }
                    history.append(action_entry)

                # Add community card deals at appropriate times
                self._add_community_card_entries(history, state)

            # Update game history regardless of whether we have decisions or not
            game['action_history_with_cards'] = history

            return game['action_history_with_cards'].copy()

        except Exception as e:
            logger.warning(f"Error building action history for session {session_id}: {str(e)}")
            logger.exception(f"📋 [ACTION_HISTORY] Exception details: {e}")
            return game.get('action_history_with_cards', []).copy()

    def _add_blind_entries(self, history, env, session_id):
        """Add small blind and big blind entries to action history"""
        try:
            # Get dealer position to determine SB/BB
            dealer_id = None
            if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                dealer_id = env.game.dealer_id

            if dealer_id is None:
                logger.warning(f"📋 [BLINDS] Could not determine dealer_id for session {session_id}")
                return

            # In heads-up poker: dealer is BB, non-dealer is SB
            sb_player_id = 1 - dealer_id  # Non-dealer is SB
            bb_player_id = dealer_id       # Dealer is BB

            # Get blind amounts
            big_blind = 2  # Default
            small_blind = 1  # Default
            if hasattr(env, 'game') and hasattr(env.game, 'big_blind'):
                big_blind = env.game.big_blind
                small_blind = big_blind // 2

            # Add small blind entry
            sb_entry = {
                'type': 'blind',
                'player_id': sb_player_id,
                'player_name': 'You' if sb_player_id == 0 else 'Opponent',
                'blind_type': 'small',
                'amount': small_blind,
                'stage': 0,  # Preflop
                'pot': small_blind,
                'hand': [],
                'public_cards': []
            }
            history.append(sb_entry)

            # Add big blind entry
            bb_entry = {
                'type': 'blind',
                'player_id': bb_player_id,
                'player_name': 'You' if bb_player_id == 0 else 'Opponent',
                'blind_type': 'big',
                'amount': big_blind,
                'stage': 0,  # Preflop
                'pot': small_blind + big_blind,
                'hand': [],
                'public_cards': []
            }
            history.append(bb_entry)

        except Exception as e:
            logger.warning(f"📋 [BLINDS] Error adding blind entries for session {session_id}: {str(e)}")

    def _add_community_card_entries(self, history, state):
        """Add community card deal entries to history"""
        raw_obs = state.get('raw_obs', {})
        stage = raw_obs.get('stage', 0)
        public_cards = raw_obs.get('public_cards', [])

        if hasattr(stage, 'value'):
            stage = stage.value

        # Add flop deal
        if stage >= 1 and len(public_cards) >= 3:
            history.insert(0, {
                'type': 'community_cards',
                'stage': 'Flop',
                'all_cards': public_cards[:3]
            })

        # Add turn deal
        if stage >= 2 and len(public_cards) >= 4:
            history.insert(0, {
                'type': 'community_cards',
                'stage': 'Turn',
                'all_cards': [public_cards[3]]
            })

        # River deal removed from action history UI
    
    def _emit_blind_events(self, session_id, env):
        """Emit blind posting events immediately after game initialization"""
        try:
            if session_id not in self.games:
                return
            
            game = self.games[session_id]
            if 'hand_events' not in game:
                game['hand_events'] = []
            
            # Get dealer position to determine SB/BB
            dealer_id = None
            if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                dealer_id = env.game.dealer_id
            
            if dealer_id is None:
                logger.warning(f"📋 [BLINDS] Could not determine dealer_id for session {session_id}")
                return
            
            # In heads-up poker: dealer is BB, non-dealer is SB
            sb_player_id = 1 - dealer_id  # Non-dealer is SB
            bb_player_id = dealer_id       # Dealer is BB
            
            # Get blind amounts
            big_blind = 2  # Default
            small_blind = 1  # Default
            if hasattr(env, 'game') and hasattr(env.game, 'big_blind'):
                big_blind = env.game.big_blind
                small_blind = big_blind // 2
            
            # Get initial pot (should be 0 before blinds)
            initial_pot = 0
            
            # Emit small blind event
            sb_event = HandEvent(
                stage='preflop',
                kind='blind',
                player_idx=sb_player_id,
                amount=small_blind,
                pot=small_blind,
                label='Post SB'
            )
            game['hand_events'].append(sb_event)
            
            # Emit big blind event
            bb_event = HandEvent(
                stage='preflop',
                kind='blind',
                player_idx=bb_player_id,
                amount=big_blind,
                pot=small_blind + big_blind,
                label='Post BB'
            )
            game['hand_events'].append(bb_event)
            
            # Mark blinds as added
            game['blinds_added'] = True
            
        except Exception as e:
            logger.warning(f"📋 [BLINDS] Error emitting blind events for session {session_id}: {str(e)}")
    
    def _emit_action_event(self, session_id, player_id, action_value, state, next_state, bet_amount=None):
        """Emit an action event when a player takes an action"""
        logger.info(f"🔍 [EMIT_ACTION_EVENT] Starting - session_id={session_id}, player_id={player_id}, action_value={action_value}")
        try:
            if session_id not in self.games:
                logger.warning(f"🔍 [EMIT_ACTION_EVENT] Session not in games, returning")
                return
            
            game = self.games[session_id]
            if 'hand_events' not in game:
                game['hand_events'] = []
            
            env = game.get('env')
            if not env:
                logger.warning(f"🔍 [EMIT_ACTION_EVENT] No env, returning")
                return
            
            # Get stage
            logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 1: Getting stage...")
            raw_obs = state.get('raw_obs', {})
            stage = raw_obs.get('stage', 0)
            logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 1a: stage (raw)={stage} (type: {type(stage)})")
            if hasattr(stage, 'value'):
                stage = stage.value
                logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 1b: stage.value={stage}")
            elif not isinstance(stage, int):
                stage = safe_int_convert(stage, 0)
                logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 1c: converted stage={stage}")
            
            stage_names = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
            stage_name = stage_names.get(stage, 'preflop')
            logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 1: Final stage_name={stage_name}")
            
            # Get pot after action
            logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 2: Getting pot...")
            next_raw_obs = next_state.get('raw_obs', {}) if next_state else {}
            pot = next_raw_obs.get('pot', raw_obs.get('pot', 0))
            logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 2a: pot (raw)={pot} (type: {type(pot)})")
            try:
                pot = float(pot) if pot else 0.0
                logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 2b: pot (converted)={pot}")
            except Exception as e:
                logger.error(f"❌ [EMIT_ACTION_EVENT] Step 2: Error converting pot: {e}")
                pot = 0.0
            
            # Get context for action labeling
            logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 3: Getting context for action labeling...")
            game_state = {'raw_obs': raw_obs}
            try:
                context = ActionLabeling.get_context_from_state(game_state, player_id, env)
                logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 3a: Got context: {context}")
            except Exception as e:
                logger.error(f"❌ [EMIT_ACTION_EVENT] Step 3: Error getting context: {e}")
                logger.error(traceback.format_exc())
                context = {}
            
            # Generate action label
            logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 4: Generating action label...")
            try:
                label = ActionLabeling.get_action_label(action_value, context, bet_amount)
                logger.info(f"🔍 [EMIT_ACTION_EVENT] Step 4a: Generated label: {label}")
            except Exception as e:
                logger.error(f"❌ [EMIT_ACTION_EVENT] Step 4: Error generating label: {e}")
                logger.error(traceback.format_exc())
                label = f"Action {action_value}"
            
            logger.info(f"📊 [EMIT_ACTION_EVENT] action_value={action_value}, bet_amount={bet_amount}, is_facing_bet={context.get('is_facing_bet', False) if context else 'N/A'}, label={label}")
            
            # Create and emit event
            event = HandEvent(
                stage=stage_name,
                kind='action',
                player_idx=player_id,
                amount=bet_amount if bet_amount else None,
                pot=pot,
                label=label,
                action_value=action_value,
                bet_amount=bet_amount
            )
            game['hand_events'].append(event)
            
        except Exception as e:
            logger.warning(f"📋 [ACTION_EVENT] Error emitting action event for session {session_id}: {str(e)}")
    
    def _emit_community_card_event(self, session_id, stage_name, cards, pot):
        """Emit a community card event when flop/turn/river are dealt"""
        try:
            if session_id not in self.games:
                return
            
            game = self.games[session_id]
            if 'hand_events' not in game:
                game['hand_events'] = []
            
            # Convert cards to string format
            card_strings = []
            for card in cards:
                if isinstance(card, str):
                    card_strings.append(card)
                else:
                    # Convert card index to string format
                    try:
                        from rlcard.games.base import Card
                        if isinstance(card, Card):
                            card_index = card.get_index()
                        else:
                            card_index = int(card)
                        
                        if 0 <= card_index <= 51:
                            suits = ['S', 'H', 'D', 'C']
                            ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
                            suit_idx = card_index // 13
                            rank_idx = card_index % 13
                            card_strings.append(suits[suit_idx] + ranks[rank_idx])
                        else:
                            card_strings.append(str(card))
                    except:
                        card_strings.append(str(card))
            
            # Create and emit event
            event = HandEvent(
                stage=stage_name.lower(),
                kind='community',
                player_idx=None,
                pot=pot,
                cards=card_strings,
                label=stage_name
            )
            game['hand_events'].append(event)
            
        except Exception as e:
            logger.warning(f"📋 [COMMUNITY_CARDS] Error emitting community card event for session {session_id}: {str(e)}")
    
    def _emit_win_event(self, session_id, winner_id, pot_bb):
        """Emit a win event when hand concludes"""
        try:
            if session_id not in self.games:
                return
            
            game = self.games[session_id]
            if 'hand_events' not in game:
                game['hand_events'] = []
            
            # Determine player name
            player_name = 'You' if winner_id == 0 else 'Opponent'
            
            # Use correct verb form based on player name
            verb = 'win' if winner_id == 0 else 'wins'
            
            # Create and emit event with formatted label
            event = HandEvent(
                stage='showdown',
                kind='win',
                player_idx=winner_id,
                amount=pot_bb,
                pot=0.0,  # Pot is distributed, so 0 remaining
                label=f'{player_name} {verb} pot of {pot_bb:.1f} BB'
            )
            game['hand_events'].append(event)
            
        except Exception as e:
            logger.warning(f"📋 [WIN_EVENT] Error emitting win event for session {session_id}: {str(e)}")
    
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
            current_stage = safe_int_convert(current_stage, 0)
        
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
        
        # Initialize raised amounts with blinds (raised array is cumulative for entire hand)
        # CRITICAL FIX: Do NOT reset raised to [0,0] for postflop - it carries over from preflop
        # The raised array represents total chips put in for the hand, not reset per betting round
        if dealer_id is not None:
            small_blind_player = (dealer_id + 1) % 2
            big_blind_player = (dealer_id + 2) % 2
            raised[small_blind_player] = small_blind
            raised[big_blind_player] = big_blind
        betting_round_start_index = 0
        
        # Replay all previous actions to get raised amounts before this action
        # previous_actions contains actions up to but not including the current action
        # action_index is the index of the current action (so we process indices 0 to action_index-1)
        logger.debug(f"🎲 [RECONSTRUCT_STATE] action_index={action_index}, previous_actions count={len(previous_actions) if previous_actions else 0}, current_public_cards_count={current_public_cards_count}, action_stage={action_stage}, initial raised={raised}")
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

                    logger.debug(f"🎲 [RECONSTRUCT_STATE] Processing prev_action i={i}: player {prev_player_id}, action {prev_action_value}, raised before={raised}")

                    # Reconstruct state at this point to calculate bet amount
                    temp_pot = calculate_pot(raised.copy(), big_blind, dealer_id, action_stage)
                    temp_state = {
                        'raw_obs': {
                            'raised': raised.copy(),
                            'big_blind': big_blind,
                            'pot': temp_pot,
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
                        logger.debug(f"🎲 [RECONSTRUCT_STATE] Raise: player {prev_player_id} raised to {raised[prev_player_id]}, bet_amount={bet_amount}")
                    elif prev_action_value == 1:  # Call/Check
                        # Need to distinguish between Check and Call
                        opponent_id = 1 - prev_player_id
                        opponent_raised = raised[opponent_id]
                        player_raised = raised[prev_player_id]

                        if action_stage == 0:  # Preflop - action 1 is always Call
                            raised[prev_player_id] = opponent_raised
                            logger.debug(f"🎲 [RECONSTRUCT_STATE] Preflop Call: player {prev_player_id} called to {raised[prev_player_id]}")
                        else:  # Postflop - action 1 can be Check or Call
                            if opponent_raised > player_raised:  # Call
                                raised[prev_player_id] = opponent_raised
                                logger.debug(f"🎲 [RECONSTRUCT_STATE] Postflop Call: player {prev_player_id} called to {raised[prev_player_id]}")
                            else:  # Check
                                # Check doesn't change raised amounts
                                logger.debug(f"🎲 [RECONSTRUCT_STATE] Postflop Check: player {prev_player_id} checked, raised unchanged at {raised[prev_player_id]}")
                    # Fold (0) and All-in (4) don't change raised amounts for the other player

                    logger.debug(f"🎲 [RECONSTRUCT_STATE] After processing: raised={raised}")
        
        # Create reconstructed state
        reconstructed_state = state.copy()
        reconstructed_state['raw_obs'] = raw_obs.copy()
        reconstructed_state['raw_obs']['raised'] = raised
        # Recalculate pot based on raised amounts using centralized calculator
        pot = calculate_pot(raised, big_blind, dealer_id, action_stage)
        reconstructed_state['raw_obs']['pot'] = pot

        logger.debug(f"🎲 [RECONSTRUCT_STATE] Final reconstructed state for action_index={action_index}: raised={raised}, pot={pot}")

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
        
        # Get stage
        current_stage = raw_obs.get('stage', 0)
        if hasattr(current_stage, 'value'):
            current_stage = current_stage.value
        elif not isinstance(current_stage, int):
            current_stage = safe_int_convert(current_stage, 0)
        
        # Get public cards count to determine betting round
        public_cards = raw_obs.get('public_cards', [])
        current_public_cards_count = len(public_cards)
        
        # Determine which betting round this action belongs to
        action_stage = 0  # Default to preflop
        if current_public_cards_count >= 5:
            action_stage = 3  # River
        elif current_public_cards_count >= 4:
            action_stage = 2  # Turn
        elif current_public_cards_count >= 3:
            action_stage = 1  # Flop
        else:
            action_stage = 0  # Preflop
        
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
                    temp_pot = calculate_pot(raised.copy(), big_blind, dealer_id, action_stage)
                    temp_state = {
                        'raw_obs': {
                            'raised': raised.copy(),
                            'big_blind': big_blind,
                            'pot': temp_pot,
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
        # Recalculate pot using centralized calculator
        pot = calculate_pot(raised, big_blind, dealer_id, action_stage)
        reconstructed_state['raw_obs']['pot'] = pot
        
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
        """Format action for display - SIMPLIFIED for native RLCard integration"""
        # SIMPLIFIED: With native RLCard state, no complex reconstruction needed
        # The state should now be consistent and accurate
        return self._action_to_string(action, env, state, player_id, bet_amount)
    
    def _action_to_string(self, action, env=None, state=None, player_id=None, bet_amount=None):
        """Convert action value to display string using shared ActionLabeling module"""
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
        
        # Use shared ActionLabeling module for consistent labeling
        try:
            if state is not None:
                context = ActionLabeling.get_context_from_state(state, player_id=player_id, env=env)
                return ActionLabeling.get_action_label(action_value, context, bet_amount=bet_amount)
        except Exception as e:
            logger.debug(f"Error using ActionLabeling, falling back to simple labels: {e}")
        
        # Fallback to simple labels if ActionLabeling fails
        if action_value == 0:
            return 'Fold'
        elif action_value == 1:
            return 'Check/Call'
        elif action_value == 2:
            return 'Raise ½ Pot'
        elif action_value == 3:
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

        # For tracking decisions, state should be next_state (after action)
        # Try to get the actual bet amount by comparing in_chips before/after action
        if state and 'raw_obs' in state and player_id is not None:
            raw_obs = state['raw_obs']

            # Try to get in_chips from the state (after action)
            in_chips = None
            if hasattr(state, 'get') and 'in_chips' in state:
                in_chips = state.get('in_chips', [0, 0])
            elif 'raw_obs' in state and 'in_chips' in state['raw_obs']:
                in_chips = state['raw_obs'].get('in_chips', [0, 0])

            # If we have in_chips, we can calculate the bet amount
            if in_chips and len(in_chips) > player_id:
                bet_amount = in_chips[player_id]
                return bet_amount

            # Fallback: try raised array (though RLCard doesn't maintain it properly)
            raised = raw_obs.get('raised', [0, 0])
            if player_id < len(raised) and raised[player_id] > 0:
                bet_amount = raised[player_id]
                return bet_amount

        # Fallback: predict bet amount for action planning (when we don't have next_state yet)

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
                        stage = safe_int_convert(stage, 0)

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
                        stage = safe_int_convert(stage, 0)
                    
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
            pot = safe_int_convert(pot, 0)
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
                        stage = safe_int_convert(stage, 0)
                    
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
                            # If facing a 3-bet (> 10BB), 4bet to 25BB total
                            elif opponent_raised > big_blind * 10:
                                return 25 * big_blind
                            # If facing a preflop raise (3BB open to ~10BB), 3bet to 10BB total
                            else:
                                return 10 * big_blind
            except Exception:
                pass
            
            return pot
        else:
            return 0
    
    def _get_opponent_hand(self, env):
        """Get opponent's hand when game is over"""
        if safe_bool_convert(env.is_over()):
            perfect_info = env.get_perfect_information()
            if 'hand_cards' in perfect_info and len(perfect_info['hand_cards']) > 1:
                return perfect_info['hand_cards'][1]
        return None
    
    def process_action(self, session_id, action_value):
        """Process a player action"""
        is_fold = (action_value == 0)
        logger.info(f"🎯 [PROCESS_ACTION] Starting action processing for {session_id}: action={action_value} {'[FOLD]' if is_fold else ''}")

        if session_id not in self.games:
            logger.warning(f"❌ [PROCESS_ACTION] Session not found: {session_id}")
            return None

        game = self.games[session_id]
        env = game['env']
        state = game['current_state']
        current_player = game['current_player']
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 1: Initial state - current_player (raw)={current_player} (type: {type(current_player)})")
        
        # Convert to int to avoid NumPy array comparison errors
        current_player = safe_int_convert(current_player, 0)
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 2: Converted current_player={current_player} (type: {type(current_player)})")
        
        human_agent = game['human_agent']

        logger.info(f"🔍 [FOLD_WORKFLOW] Step 3: Game state - current_player={current_player}, env_type={type(env)}, action_value={action_value}")

        # Edge case: Game already over
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 4: Checking if game is over...")
        try:
            is_over_raw = env.is_over()
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 4a: env.is_over() raw result: {is_over_raw} (type: {type(is_over_raw)})")
            game_over = safe_bool_convert(is_over_raw)
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 4b: Converted game_over: {game_over} (type: {type(game_over)})")
            if game_over:
                logger.warning(f"🏁 [FOLD_WORKFLOW] Game already over, returning game state")
                return self.get_game_state(session_id)
        except Exception as e:
            logger.error(f"❌ [FOLD_WORKFLOW] Error checking if game is over: {e}")
            logger.error(traceback.format_exc())
            return None

        # Edge case: Not player's turn
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 5: Checking if it's player's turn... current_player={current_player}")
        if current_player != 0:
            logger.warning(f"❌ [FOLD_WORKFLOW] Not player's turn (current: P{current_player})")
            return self.get_game_state(session_id)
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 5: ✓ It's player's turn")
        
        # Get legal actions - try raw_legal_actions first, fallback to legal_actions dict
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 6: Getting legal actions from state...")
        raw_legal_actions = state['raw_obs'].get('raw_legal_actions', [])
        legal_actions_dict = state['raw_obs'].get('legal_actions', {})
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 6a: raw_legal_actions={raw_legal_actions} (type: {type(raw_legal_actions)})")
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 6b: legal_actions_dict={legal_actions_dict} (type: {type(legal_actions_dict)})")

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

        # Convert legal_actions dict to list if needed
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 7: Converting legal actions...")
        legal_actions_list = []
        if raw_legal_actions and len(raw_legal_actions) > 0:
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 7a: Using raw_legal_actions, converting...")
            legal_actions_list = convert_actions(raw_legal_actions)  # Convert Action enums to integers
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 7a: Converted legal_actions_list={legal_actions_list}")
        elif legal_actions_dict:
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 7b: Using legal_actions_dict, converting...")
            # Convert dict keys to list
            if isinstance(legal_actions_dict, dict):
                legal_actions_list = convert_actions(list(legal_actions_dict.keys()))
            elif isinstance(legal_actions_dict, list):
                legal_actions_list = convert_actions(legal_actions_dict)
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 7b: Converted legal_actions_list={legal_actions_list}")
        else:
            logger.warning(f"🔍 [FOLD_WORKFLOW] Step 7c: No legal actions found in state!")

        # Get the TRUE legal actions directly from RLCard environment
        # This is critical because RLCard's env.step() will only accept actions that env.get_legal_actions() returns
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 8: Getting RLCard legal actions from environment...")
        try:
            rlcard_legal_actions = env.get_legal_actions()
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 8a: env.get_legal_actions() returned: {rlcard_legal_actions} (type: {type(rlcard_legal_actions)})")
            original_legal_actions = convert_actions(rlcard_legal_actions)
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 8b: Converted original_legal_actions={original_legal_actions}")
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 8c: Checking if fold (0) is in legal actions: {0 in original_legal_actions}")
        except Exception as e:
            logger.warning(f"⚠️ [FOLD_WORKFLOW] Step 8: RLCard legal actions failed ({e}), using fallback")
            logger.warning(traceback.format_exc())
            # If RLCard doesn't support get_legal_actions, fall back to basic poker actions
            # But ensure we have valid actions - don't rely on potentially empty state data
            if legal_actions_list and len(legal_actions_list) > 0:
                original_legal_actions = legal_actions_list.copy()
                logger.info(f"🔍 [FOLD_WORKFLOW] Step 8: Using state legal actions: {original_legal_actions}")
            else:
                # Ultimate fallback: basic poker actions for current stage
                raw_obs = state.get('raw_obs', {})
                stage = raw_obs.get('stage', 0)
                if stage == 0:  # Preflop
                    original_legal_actions = [0, 1, 3, 4]  # FOLD, CHECK_CALL, RAISE_POT, ALL_IN (avoid RAISE_HALF_POT issues)
                else:  # Postflop
                    original_legal_actions = [0, 1, 2, 3, 4]  # All actions
                logger.info(f"🔍 [FOLD_WORKFLOW] Step 8: Using fallback legal actions for stage {stage}: {original_legal_actions}")

        # Get stage and button_id for the fix (same logic as in get_game_state)
        raw_obs = state.get('raw_obs', {})
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        elif not isinstance(stage, int):
            stage = safe_int_convert(stage, 0)

        # Get dealer_id from game object (not in raw_obs)
        dealer_id = None
        if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
            dealer_id = env.game.dealer_id
            logger.debug(f"🔍 [PROCESS_ACTION] Raw dealer_id: {dealer_id} (type: {type(dealer_id)})")
            # Convert to int to avoid NumPy array comparison errors
            dealer_id = safe_int_convert(dealer_id, None)
            logger.debug(f"🔍 [PROCESS_ACTION] Converted dealer_id: {dealer_id} (type: {type(dealer_id)})")

        # Calculate button_id (small blind in heads-up)
        button_id = None
        if dealer_id is not None:
            button_id = (dealer_id + 1) % 2  # In HUNL, button = (dealer + 1) % 2
            logger.debug(f"🔍 [PROCESS_ACTION] Raw button_id: {button_id} (type: {type(button_id)})")
            # Convert to int to avoid NumPy array comparison errors
            button_id = safe_int_convert(button_id, None)
            logger.debug(f"🔍 [PROCESS_ACTION] Converted button_id: {button_id} (type: {type(button_id)})")

        # Fix for small blind first to act preflop - ensure standard raise options are available
        # This must match the fix in get_game_state to keep indices consistent
        artificially_added_actions = []
        
        # Add detailed logging for condition checks to debug NumPy array comparison issues
        logger.debug(f"🔍 [PROCESS_ACTION] Condition check - stage: {stage} (type: {type(stage)}), current_player: {current_player} (type: {type(current_player)}), button_id: {button_id} (type: {type(button_id)})")
        logger.debug(f"🔍 [PROCESS_ACTION] legal_actions_list: {legal_actions_list} (types: {[type(x) for x in legal_actions_list[:5]] if legal_actions_list else []})")
        logger.debug(f"🔍 [PROCESS_ACTION] original_legal_actions: {original_legal_actions} (types: {[type(x) for x in original_legal_actions[:5]] if original_legal_actions else []})")
        
        try:
            stage_check = stage == 0
            logger.debug(f"🔍 [PROCESS_ACTION] stage == 0: {stage_check} (type: {type(stage_check)})")
        except Exception as e:
            logger.error(f"❌ [PROCESS_ACTION] Error comparing stage == 0: {e}, stage={stage}, type={type(stage)}")
            raise
        
        try:
            player_check = current_player == 0
            logger.debug(f"🔍 [PROCESS_ACTION] current_player == 0: {player_check} (type: {type(player_check)})")
        except Exception as e:
            logger.error(f"❌ [PROCESS_ACTION] Error comparing current_player == 0: {e}, current_player={current_player}, type={type(current_player)}")
            raise
        
        try:
            button_check = button_id == 0 if button_id is not None else False
            logger.debug(f"🔍 [PROCESS_ACTION] button_id == 0: {button_check} (type: {type(button_check)})")
        except Exception as e:
            logger.error(f"❌ [PROCESS_ACTION] Error comparing button_id == 0: {e}, button_id={button_id}, type={type(button_id)}")
            raise
        
        if (stage == 0 and  # Preflop
            current_player == 0 and  # Human player (small blind in heads-up)
            not safe_bool_convert(env.is_over()) and
            button_id == 0):  # Small blind is first to act

            # Small blind should always be able to make standard preflop raises
            # RLCard sometimes incorrectly marks RAISE_HALF_POT as illegal
            try:
                action2_in_list = 2 in legal_actions_list
                logger.debug(f"🔍 [PROCESS_ACTION] 2 in legal_actions_list: {action2_in_list} (type: {type(action2_in_list)})")
            except Exception as e:
                logger.error(f"❌ [PROCESS_ACTION] Error checking '2 in legal_actions_list': {e}, legal_actions_list={legal_actions_list}, types={[type(x) for x in legal_actions_list] if legal_actions_list else []}")
                raise
            
            try:
                action3_in_original = 3 in original_legal_actions
                logger.debug(f"🔍 [PROCESS_ACTION] 3 in original_legal_actions: {action3_in_original} (type: {type(action3_in_original)})")
            except Exception as e:
                logger.error(f"❌ [PROCESS_ACTION] Error checking '3 in original_legal_actions': {e}, original_legal_actions={original_legal_actions}, types={[type(x) for x in original_legal_actions] if original_legal_actions else []}")
                raise
            
            logger.info(f"🔧 [PROCESS_ACTION] Small blind preflop check - Action 2 in list: {action2_in_list}, Action 3 in RLCard: {action3_in_original}")
            if 2 not in legal_actions_list and 3 in original_legal_actions:  # RAISE_HALF_POT (Raise to 3 BB) only if RAISE_POT is available in RLCard
                legal_actions_list.append(2)
                artificially_added_actions.append(2)
                logger.info(f"🔧 [PROCESS_ACTION] Added artificial action 2 (RAISE_HALF_POT) to legal actions list")
        
        # Edge case: No legal actions
        if not legal_actions_list or len(legal_actions_list) == 0:
            logger.warning(f"No legal actions available for session {session_id}")
            return self.get_game_state(session_id)
        
        # Handle artificially added actions that RLCard doesn't actually support
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 9: Processing action mapping...")
        actual_action_value = action_value
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 9a: action_value={action_value}, artificially_added_actions={artificially_added_actions}")
        
        try:
            action_in_artificial = action_value in artificially_added_actions
            logger.debug(f"🔍 [PROCESS_ACTION] action_value in artificially_added_actions: {action_in_artificial} (type: {type(action_in_artificial)})")
        except Exception as e:
            logger.error(f"❌ [PROCESS_ACTION] Error checking 'action_value in artificially_added_actions': {e}, action_value={action_value}, artificially_added_actions={artificially_added_actions}")
            raise

        if action_value in artificially_added_actions:
            # Map artificially added actions to legal alternatives
            try:
                action3_in_original = 3 in original_legal_actions
                logger.debug(f"🔍 [PROCESS_ACTION] 3 in original_legal_actions (artificial check): {action3_in_original} (type: {type(action3_in_original)})")
            except Exception as e:
                logger.error(f"❌ [PROCESS_ACTION] Error checking '3 in original_legal_actions' (artificial): {e}, original_legal_actions={original_legal_actions}")
                raise
            
            if action_value == 2 and 3 in original_legal_actions:  # RAISE_HALF_POT -> RAISE_POT
                actual_action_value = 3
            else:
                logger.error(f'❌ [PROCESS_ACTION] Action {action_value} has no legal alternative in {original_legal_actions}')
                raise ValueError(f'Action {action_value} was artificially added but no legal alternative found')
        elif action_value == 2:
            # Special case: Action 2 (RAISE_HALF_POT) may be sent even if not artificially added
            # If action 3 (RAISE_POT) is legal in RLCard, map action 2 to action 3
            try:
                action3_in_original = 3 in original_legal_actions
                action2_in_original = 2 in original_legal_actions
                logger.debug(f"🔍 [PROCESS_ACTION] 3 in original_legal_actions (action 2 check): {action3_in_original}, 2 in original_legal_actions: {action2_in_original}")
            except Exception as e:
                logger.error(f"❌ [PROCESS_ACTION] Error checking actions in original_legal_actions (action 2): {e}, original_legal_actions={original_legal_actions}")
                raise
            
            if 3 in original_legal_actions:
                actual_action_value = 3
            elif 2 in original_legal_actions:
                # Action 2 is actually legal in RLCard, use it directly
                actual_action_value = 2
            else:
                logger.error(f'❌ [PROCESS_ACTION] Action {action_value} not legal in {original_legal_actions}')
                raise ValueError(f'Action {action_value} is not legal in RLCard and cannot be mapped')

        # Find the action index, handling both integers and Action enums
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 10: Finding action index in original_legal_actions...")
        action_index = None
        action = None
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 10a: actual_action_value={actual_action_value} (type: {type(actual_action_value)}), original_legal_actions={original_legal_actions} (count: {len(original_legal_actions)})")
        
        for i, orig_action in enumerate(original_legal_actions):
            try:
                if hasattr(orig_action, 'value'):
                    orig_action_val = orig_action.value
                    logger.debug(f"🔍 [PROCESS_ACTION] Comparing action {i}: orig_action.value={orig_action_val} (type: {type(orig_action_val)}) with actual_action_value={actual_action_value} (type: {type(actual_action_value)})")
                    try:
                        comparison_result = actual_action_value == orig_action_val
                        logger.debug(f"🔍 [PROCESS_ACTION] Comparison result: {comparison_result} (type: {type(comparison_result)})")
                        if comparison_result:
                            action_index = i
                            action = orig_action
                            break
                    except Exception as e:
                        logger.error(f"❌ [PROCESS_ACTION] Error comparing actual_action_value == orig_action.value: {e}, actual_action_value={actual_action_value}, orig_action_val={orig_action_val}")
                        raise
                else:
                    logger.debug(f"🔍 [PROCESS_ACTION] Comparing action {i}: orig_action={orig_action} (type: {type(orig_action)}) with actual_action_value={actual_action_value} (type: {type(actual_action_value)})")
                    try:
                        comparison_result = actual_action_value == orig_action
                        logger.debug(f"🔍 [PROCESS_ACTION] Comparison result: {comparison_result} (type: {type(comparison_result)})")
                        if comparison_result:
                            action_index = i
                            action = orig_action
                            break
                    except Exception as e:
                        logger.error(f"❌ [PROCESS_ACTION] Error comparing actual_action_value == orig_action: {e}, actual_action_value={actual_action_value}, orig_action={orig_action}")
                        raise
            except Exception as e:
                logger.error(f"❌ [PROCESS_ACTION] Error in action finding loop at index {i}: {e}")
                raise

        if action_index is None:
            # Special debug: check what we have
            logger.error(f"❌ [FOLD_WORKFLOW] Step 10: Could not find action index!")
            action_values_in_list = []
            for orig_action in original_legal_actions:
                if hasattr(orig_action, 'value'):
                    action_values_in_list.append(orig_action.value)
                else:
                    action_values_in_list.append(orig_action)
            logger.error(f"❌ [FOLD_WORKFLOW] Step 10: Available values: {action_values_in_list}, Full RLCard actions: {original_legal_actions}")
            raise ValueError(f'Could not find action {actual_action_value} in RLCard legal actions. Available: {action_values_in_list}, RLCard legal actions: {original_legal_actions}')

        logger.info(f"🔍 [FOLD_WORKFLOW] Step 10b: ✓ Found action_index={action_index}, action={action}")

        # Edge case: Action is None or invalid
        if action is None:
            logger.error(f"❌ [FOLD_WORKFLOW] Step 11: Action is None!")
            raise ValueError('Action is None')
        
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 11: ✓ Action is valid: {action}")

        # Get action details for logging
        raw_obs = state.get('raw_obs', {})
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        elif not isinstance(stage, int):
            stage = safe_int_convert(stage, 0)

        pot = raw_obs.get('pot', 0)
        public_cards = raw_obs.get('public_cards', [])
        player_hand = game.get('player_hand', [])

        stage_names = {0: 'Preflop', 1: 'Flop', 2: 'Turn', 3: 'River'}
        stage_name = stage_names.get(stage, f'Stage {stage}')
        action_name = self._action_to_string(action, env, state, current_player)

        # Log what user requested vs what was actually executed
        requested_action_name = self._action_to_string(action_value, env, state, current_player)
        if action_value != actual_action_value:
            logger.info(f"👤 Player action - Session: {session_id}, {stage_name}, Requested: {requested_action_name} ({action_value}) -> Executed: {action_name} ({action}), Pot: {pot}, Cards: {len(public_cards)}, Hand: {player_hand}")
        else:
            logger.info(f"👤 Player action - Session: {session_id}, {stage_name}, {action_name} ({action}), Pot: {pot}, Cards: {len(public_cards)}, Hand: {player_hand}")
        raised = raw_obs.get('raised', [0, 0])

        # CRITICAL FIX: Re-check legal actions right before env.step() to ensure they haven't changed
        # This fixes the "Action not allowed" error that can occur due to timing issues
        try:
            current_legal_actions = env.get_legal_actions()
            current_legal_actions = convert_actions(current_legal_actions)
            logger.info(f"🔍 [PROCESS_ACTION] Re-checking legal actions before env.step(): {current_legal_actions}")

            # Verify our action_index is still valid
            if action_index >= len(current_legal_actions):
                logger.error(f"❌ [PROCESS_ACTION] Action index {action_index} is out of bounds for current legal actions: {current_legal_actions}")
                raise ValueError(f'Action index {action_index} is no longer valid. Current legal actions: {current_legal_actions}')

            current_action_at_index = current_legal_actions[action_index]
            # Extract value from action (handle both Action enum and int)
            current_action_value = current_action_at_index.value if hasattr(current_action_at_index, 'value') else current_action_at_index
            if current_action_value != actual_action_value:
                logger.warning(f"⚠️ [PROCESS_ACTION] Action at index {action_index} changed from {actual_action_value} to {current_action_value}. Using new value.")
                actual_action_value = current_action_value
                # Re-find the action index with the updated value
                for i, orig_action in enumerate(original_legal_actions):
                    if hasattr(orig_action, 'value'):
                        if actual_action_value == orig_action.value:
                            action_index = i
                            action = orig_action
                            break
                    else:
                        if actual_action_value == orig_action:
                            action_index = i
                            action = orig_action
                            break
        except AttributeError:
            # RLCard environment doesn't have get_legal_actions() method, skip the re-check
            logger.info(f"🔍 [PROCESS_ACTION] Skipping legal actions re-check (RLCard doesn't support it)")

        # SIMPLIFIED: Hand capture no longer needed - _track_decision gets it from RLCard state directly

        # Set action in human agent
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 12: Setting action in human agent...")
        human_agent.set_action(action)
        logger.info(f"🔍 [FOLD_WORKFLOW] Step 12: ✓ Action set in human agent")

        try:
            # Use raw_action=False mode which does validation and conversion through _decode_action
            step_action = action  # Pass the action value (int) for _decode_action to convert to Action enum
            raw_action = False  # Let RLCard decode and validate the action

            logger.info(f"🔍 [FOLD_WORKFLOW] Step 13: Calling env.step() with step_action={step_action} (type: {type(step_action)}), raw_action={raw_action}")

            try:
                logger.info(f"🔍 [FOLD_WORKFLOW] Step 13a: About to call env.step()...")
                next_state, next_player_id = env.step(step_action, raw_action)
                logger.info(f"🔍 [FOLD_WORKFLOW] Step 13b: ✓ env.step() returned successfully")
                logger.info(f"🔍 [FOLD_WORKFLOW] Step 13c: next_state type: {type(next_state)}, next_player_id={next_player_id} (type: {type(next_player_id)})")
            except Exception as step_error:
                logger.error(f"❌ [FOLD_WORKFLOW] Step 13: env.step() failed: {step_error}")
                logger.error(traceback.format_exc())
                raise step_error

            # Edge case: Environment error
            if next_state is None:
                logger.error(f"❌ [PROCESS_ACTION] Environment returned None state for session {session_id}, action: {action}")
                # Try to return current game state as fallback
                try:
                    logger.info(f"🔄 [PROCESS_ACTION] Attempting fallback to current game state")
                    return self.get_game_state(session_id)
                except Exception as fallback_error:
                    logger.error(f"❌ [PROCESS_ACTION] Failed to get game state as fallback: {fallback_error}")
                    return None

            # REMOVED: BB-first action order override logic (60+ lines)
            # This functionality is now handled natively by the modified RLCard fork
            # The fork implements BB-first postflop action order directly in the game engine

            # Update game state - use RLCard's natural next_player_id
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 14: Updating game state...")
            game['current_state'] = next_state
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 14a: Updated current_state")
            
            # Check if game is over after the step
            try:
                is_over = safe_bool_convert(env.is_over())
                logger.info(f"🔍 [FOLD_WORKFLOW] Step 14: Game over check after step: {is_over}")
                
                if is_over:
                    # Game ended (e.g., player folded in heads-up)
                    # Don't update current_player to None - keep it as the player who just acted
                    # This prevents "Not player's turn" errors when the frontend checks game state
                    logger.info(f"🔍 [FOLD_WORKFLOW] Step 14: Game is over, keeping current_player={current_player}")
                    # current_player remains unchanged
                else:
                    # Game continues - update to next player
                    next_player_id_safe = safe_int_convert(next_player_id, None)
                    logger.info(f"🔍 [FOLD_WORKFLOW] Step 14b: next_player_id (raw)={next_player_id} (type: {type(next_player_id)}), converted={next_player_id_safe}")
                    if next_player_id_safe is not None:
                        game['current_player'] = next_player_id_safe
                        logger.info(f"🔍 [FOLD_WORKFLOW] Step 14c: Updated current_player={next_player_id_safe}")
                    else:
                        logger.warning(f"⚠️ [FOLD_WORKFLOW] Step 14: next_player_id is None but game is not over, keeping current_player={current_player}")
            except Exception as e:
                logger.error(f"❌ [FOLD_WORKFLOW] Step 14: Error checking if game is over: {e}")
                # Fallback: try to update current_player if next_player_id is valid
                next_player_id_safe = safe_int_convert(next_player_id, None)
                if next_player_id_safe is not None:
                    game['current_player'] = next_player_id_safe
                    logger.info(f"🔍 [FOLD_WORKFLOW] Step 14c (fallback): Updated current_player={next_player_id_safe}")
            
            # Get updated state info for logging
            next_raw_obs = next_state.get('raw_obs', {}) if next_state else {}
            next_pot = next_raw_obs.get('pot', 0)
            next_public_cards = next_raw_obs.get('public_cards', [])
            next_raised = next_raw_obs.get('raised', [0, 0])
            
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 15: State after action - next_pot={next_pot}, next_public_cards count={len(next_public_cards)}, next_raised={next_raised}")
            
            # Log action result
            current_player_after = game.get('current_player', 'Unknown')
            logger.info(f"✅ [FOLD_WORKFLOW] Action processed - Session: {session_id}, Next: P{current_player_after}, Pot: {next_pot}, Cards: {len(next_public_cards)}")

            # Track decision for hand history
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 16: Tracking decision - Player: {current_player}, Action: {action}")

            # Update state's current_player to reflect the player who just acted
            state['raw_obs']['current_player'] = current_player

            # Calculate bet amount for event emission
            bet_amount = None
            try:
                # Get raised amounts before and after action
                previous_raised = raw_obs.get('raised', [0, 0])
                next_raised = next_raw_obs.get('raised', [0, 0])
                
                logger.debug(f"🔍 [BET_AMOUNT] previous_raised: {previous_raised} (types: {[type(x) for x in previous_raised] if isinstance(previous_raised, list) else type(previous_raised)})")
                logger.debug(f"🔍 [BET_AMOUNT] next_raised: {next_raised} (types: {[type(x) for x in next_raised] if isinstance(next_raised, list) else type(next_raised)})")
                logger.debug(f"🔍 [BET_AMOUNT] action_value: {action_value} (type: {type(action_value)})")
                logger.debug(f"🔍 [BET_AMOUNT] current_player: {current_player} (type: {type(current_player)})")
                
                if action_value in [2, 3, 4]:  # Raise actions
                    # For raises, bet_amount is the total amount raised TO
                    if current_player == 0:
                        bet_amount = next_raised[0] if len(next_raised) > 0 else 0
                        logger.debug(f"🔍 [BET_AMOUNT] Raise - Player 0, bet_amount: {bet_amount} (type: {type(bet_amount)})")
                    else:
                        bet_amount = next_raised[1] if len(next_raised) > 1 else 0
                        logger.debug(f"🔍 [BET_AMOUNT] Raise - Player 1, bet_amount: {bet_amount} (type: {type(bet_amount)})")
                elif action_value == 1:  # Check/Call
                    # For Check/Call, calculate the amount needed to call using PREVIOUS state
                    # After a call, raised array equalizes, so we need to use before-state
                    if current_player == 0:
                        opponent_raised_before = previous_raised[1] if len(previous_raised) > 1 else 0
                        player_raised_before = previous_raised[0] if len(previous_raised) > 0 else 0
                        logger.debug(f"🔍 [BET_AMOUNT] Player 0 - opponent_raised_before: {opponent_raised_before} (type: {type(opponent_raised_before)}), player_raised_before: {player_raised_before} (type: {type(player_raised_before)})")
                        
                        # Convert to Python native types to avoid NumPy array comparison issues
                        opponent_raised_before = safe_int_convert(opponent_raised_before, 0)
                        player_raised_before = safe_int_convert(player_raised_before, 0)
                        
                        call_amount = opponent_raised_before - player_raised_before
                        logger.debug(f"🔍 [BET_AMOUNT] Player 0 - call_amount: {call_amount} (type: {type(call_amount)})")
                        
                        try:
                            call_amount_gt_zero = call_amount > 0
                            logger.debug(f"🔍 [BET_AMOUNT] Player 0 - call_amount > 0: {call_amount_gt_zero} (type: {type(call_amount_gt_zero)})")
                        except Exception as e:
                            logger.error(f"❌ [BET_AMOUNT] Error comparing call_amount > 0: {e}, call_amount={call_amount}, type={type(call_amount)}")
                            raise
                        
                        bet_amount = call_amount if call_amount > 0 else 0
                        bet_amount = safe_int_convert(bet_amount, 0)
                        logger.debug(f"📊 [CALL_CALC] Player 0: opponent_raised={opponent_raised_before}, player_raised={player_raised_before}, call_amount={call_amount}, bet_amount={bet_amount}")
                    else:
                        opponent_raised_before = previous_raised[0] if len(previous_raised) > 0 else 0
                        player_raised_before = previous_raised[1] if len(previous_raised) > 1 else 0
                        logger.debug(f"🔍 [BET_AMOUNT] Player 1 - opponent_raised_before: {opponent_raised_before} (type: {type(opponent_raised_before)}), player_raised_before: {player_raised_before} (type: {type(player_raised_before)})")
                        
                        # Convert to Python native types to avoid NumPy array comparison issues
                        opponent_raised_before = safe_int_convert(opponent_raised_before, 0)
                        player_raised_before = safe_int_convert(player_raised_before, 0)
                        
                        call_amount = opponent_raised_before - player_raised_before
                        logger.debug(f"🔍 [BET_AMOUNT] Player 1 - call_amount: {call_amount} (type: {type(call_amount)})")
                        
                        try:
                            call_amount_gt_zero = call_amount > 0
                            logger.debug(f"🔍 [BET_AMOUNT] Player 1 - call_amount > 0: {call_amount_gt_zero} (type: {type(call_amount_gt_zero)})")
                        except Exception as e:
                            logger.error(f"❌ [BET_AMOUNT] Error comparing call_amount > 0: {e}, call_amount={call_amount}, type={type(call_amount)}")
                            raise
                        
                        bet_amount = call_amount if call_amount > 0 else 0
                        bet_amount = safe_int_convert(bet_amount, 0)
                        logger.debug(f"📊 [CALL_CALC] Player 1: opponent_raised={opponent_raised_before}, player_raised={player_raised_before}, call_amount={call_amount}, bet_amount={bet_amount}")
            except Exception as e:
                logger.error(f"❌ [BET_AMOUNT] Error calculating bet_amount for player event: {e}")
                logger.error(traceback.format_exc())
                bet_amount = None

            # Emit action event
            logger.debug(f"🔍 [PROCESS_ACTION] About to call _emit_action_event - current_player: {current_player} (type: {type(current_player)}), action_value: {action_value} (type: {type(action_value)}), bet_amount: {bet_amount} (type: {type(bet_amount)})")
            try:
                self._emit_action_event(session_id, current_player, action_value, state, next_state, bet_amount)
                logger.debug(f"✅ [PROCESS_ACTION] _emit_action_event completed successfully")
            except Exception as e:
                logger.error(f"❌ [PROCESS_ACTION] Error in _emit_action_event: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Also track decision for legacy compatibility
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 17: Calling _track_decision - current_player: {current_player} (type: {type(current_player)}), action: {action} (type: {type(action)})")
            try:
                self._track_decision(session_id, current_player, action, state, next_state)
                logger.info(f"🔍 [FOLD_WORKFLOW] Step 17: ✓ _track_decision completed successfully")
            except Exception as e:
                logger.error(f"❌ [FOLD_WORKFLOW] Step 17: Error in _track_decision: {e}")
                logger.error(traceback.format_exc())
                raise

            logger.info(f"🔍 [FOLD_WORKFLOW] Step 18: Getting final game state and returning...")
            final_state = self.get_game_state(session_id)
            logger.info(f"🔍 [FOLD_WORKFLOW] Step 18: ✓ Got final game state, returning")
            return final_state
        except Exception as e:
            logger.error(f"❌ [FOLD_WORKFLOW] ERROR in process_action for session {session_id}: {str(e)}")
            logger.error(f"❌ [FOLD_WORKFLOW] Error type: {type(e).__name__}")
            logger.error(f"❌ [FOLD_WORKFLOW] Action details - Requested: {action_value}, Mapped: {actual_action_value if 'actual_action_value' in locals() else 'N/A'}, Index: {action_index if 'action_index' in locals() else 'N/A'}, Action: {action if 'action' in locals() else 'N/A'}")
            logger.error(f"❌ [FOLD_WORKFLOW] Legal actions context - RLCard: {original_legal_actions if 'original_legal_actions' in locals() else 'N/A'}, Modified: {legal_actions_list if 'legal_actions_list' in locals() else 'N/A'}, Artificial: {artificially_added_actions if 'artificially_added_actions' in locals() else 'N/A'}")
            logger.error(f"❌ [FOLD_WORKFLOW] Full traceback:")
            logger.error(traceback.format_exc())
            raise
    
    def process_ai_turn(self, session_id):
        """Process AI opponent's turn"""
        logger.info(f"🤖 [AI_TURN] Starting AI turn processing for {session_id}")

        if session_id not in self.games:
            logger.warning(f"❌ [AI_TURN] Game session not found for AI turn: {session_id}. Available sessions: {list(self.games.keys())}")
            return None

        game = self.games[session_id]
        env = game['env']
        state = game['current_state']
        current_player = game['current_player']
        # Convert to int to avoid NumPy array comparison errors
        current_player = safe_int_convert(current_player, 0)
        ai_agent = game['ai_agent']

        logger.debug(f"🤖 [AI_TURN] Game state: current_player={current_player}, env_type={type(env)}, ai_agent_type={type(ai_agent)}")

        # Edge case: Not AI's turn
        if current_player != 1:
            logger.warning(f"❌ [AI_TURN] Not AI's turn (current_player={current_player}) for session {session_id}")
            return self.get_game_state(session_id)

        # Edge case: Game already over
        try:
            game_over = safe_bool_convert(env.is_over())
            logger.debug(f"🤖 [AI_TURN] Game over check: {game_over}")
            if game_over:
                logger.info(f"🏁 [AI_TURN] Game already over for session {session_id}")
                return self.get_game_state(session_id)
        except Exception as e:
            logger.error(f"❌ [AI_TURN] Error checking if game is over: {e}")
            logger.error(traceback.format_exc())
            return None
        
        try:
            # Add dealer_id to state for position determination
            dealer_id = None
            if hasattr(env, 'game') and hasattr(env.game, 'dealer_id'):
                dealer_id = env.game.dealer_id
            state_with_dealer = state.copy()
            state_with_dealer['dealer_id'] = dealer_id

            # Pass actual action history to LLM agent (if available)
            # This ensures LLM knows about all previous actions, especially raises
            if 'action_history_with_cards' in game:
                state_with_dealer['action_history_with_cards'] = game['action_history_with_cards'].copy()
                logger.info(f"📋 [AI_TURN] Passing {len(state_with_dealer['action_history_with_cards'])} action history entries to LLM agent")

            # Ensure legal actions are available in state for AI agent
            if not state_with_dealer.get('raw_legal_actions'):
                try:
                    env_legal_actions = env.get_legal_actions()
                    state_with_dealer['raw_legal_actions'] = env_legal_actions
                    logger.info(f"✅ [AI_TURN] Added raw_legal_actions from env: {env_legal_actions}")
                except Exception as e:
                    logger.warning(f"⚠️ [AI_TURN] Failed to get legal actions from env: {e}")
                    # Fallback to basic actions based on game stage
                    raw_obs = state_with_dealer.get('raw_obs', {})
                    stage = raw_obs.get('stage', 0)
                    if stage == 0:  # Preflop
                        fallback_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_POT, Action.ALL_IN]
                    else:  # Postflop
                        fallback_actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
                    state_with_dealer['raw_legal_actions'] = fallback_actions
                    logger.info(f"🔄 [AI_TURN] Using fallback legal actions for stage {stage}: {fallback_actions}")

            logger.info(f"🔍 [AI_TURN] State being passed to AI agent: raw_legal_actions = {state_with_dealer.get('raw_legal_actions', [])}")

            # Get AI action
            logger.debug(f"🤖 [AI_TURN] Calling ai_agent.eval_step() with ai_agent type: {type(ai_agent)}")
            try:
                action, _ = ai_agent.eval_step(state_with_dealer)
                logger.debug(f"✅ [AI_TURN] AI agent returned action: {action} (type: {type(action)})")
            except Exception as ai_error:
                logger.error(f"❌ [AI_TURN] AI agent eval_step failed: {ai_error}")
                logger.error(traceback.format_exc())
                return None

            # Edge case: AI action is None or invalid
            if action is None:
                logger.warning(f"❌ [AI_TURN] AI agent returned None action for session {session_id}")
                # Default to fold if action is None
                legal_actions = state_with_dealer.get('raw_legal_actions', [])
                if legal_actions and len(legal_actions) > 0:
                    action = legal_actions[0]  # First action is usually fold/check
                    logger.info(f"🔄 [AI_TURN] Using fallback action: {action}")
                else:
                    logger.error(f"❌ [AI_TURN] No legal actions available for fallback")
                    return self.get_game_state(session_id)

            # Extract action_value as a Python int (not NumPy array) to avoid boolean ambiguity errors
            # This must be done before any comparisons to ensure it's a scalar
            if hasattr(action, 'value'):
                action_value_raw = action.value
            else:
                action_value_raw = action
            
            # Convert to Python int if it's a NumPy type to avoid "truth value of array" errors
            if isinstance(action_value_raw, np.ndarray):
                action_value = int(action_value_raw.item()) if action_value_raw.size == 1 else int(action_value_raw[0])
            elif isinstance(action_value_raw, (np.integer, np.floating)):
                action_value = int(action_value_raw)
            else:
                action_value = int(action_value_raw) if action_value_raw is not None else 0

            # Find action index for use_raw=True agents
            action_index = None
            if ai_agent.use_raw:
                # Get legal actions from the state that was passed to the AI agent
                legal_actions = state_with_dealer.get('raw_legal_actions', [])
                logger.info(f"🔍 [AI_TURN] Finding action index for {action} in legal actions: {legal_actions}")
                logger.info(f"🔍 [AI_TURN] Full raw_obs: {state_with_dealer.get('raw_obs', {})}")
                for i, legal_action in enumerate(legal_actions):
                    # Handle both Action enums and integers
                    legal_value = legal_action.value if hasattr(legal_action, 'value') else legal_action
                    # Convert legal_value to int if it's a NumPy type
                    if isinstance(legal_value, np.ndarray):
                        legal_value = int(legal_value.item()) if legal_value.size == 1 else int(legal_value[0])
                    elif isinstance(legal_value, (np.integer, np.floating)):
                        legal_value = int(legal_value)
                    else:
                        legal_value = int(legal_value) if legal_value is not None else 0
                    
                    logger.debug(f"🔍 [AI_TURN] Checking legal action {i}: {legal_action} (value: {legal_value}) vs action {action} (value: {action_value})")
                    if legal_value == action_value:
                        action_index = i
                        break
                if action_index is None:
                    logger.error(f"❌ [AI_TURN] Could not find action {action} in legal actions {legal_actions}")
                    logger.error(f"❌ [AI_TURN] Action type: {type(action)}, Action value: {action_value}")
                    logger.error(f"❌ [AI_TURN] Available legal actions values: {[a.value if hasattr(a, 'value') else a for a in legal_actions]}")
                    return self.get_game_state(session_id)
            
            # Get action details for logging
            raw_obs = state.get('raw_obs', {})
            stage = raw_obs.get('stage', 0)
            if hasattr(stage, 'value'):
                stage = stage.value
            elif not isinstance(stage, int):
                stage = safe_int_convert(stage, 0)
            
            stage_names = {0: 'Preflop', 1: 'Flop', 2: 'Turn', 3: 'River'}
            stage_name = stage_names.get(stage, f'Stage {stage}')
            action_name = self._action_to_string(action, env, state, current_player)
            pot = raw_obs.get('pot', 0)
            public_cards = raw_obs.get('public_cards', [])
            raised = raw_obs.get('raised', [0, 0])

            # SIMPLIFIED: Hand capture no longer needed - _track_decision gets it from RLCard state directly

            # Process action - pass Action enum value when use_raw=True, action_index when use_raw=False
            # Note: RLCard's use_raw=True means "use Action enum directly", not "use indices"
            if ai_agent.use_raw:
                step_action = action  # Pass the Action enum value
            else:
                step_action = action_index  # Pass the index into legal actions
            logger.debug(f"🤖 [AI_TURN] Calling env.step({step_action}, {ai_agent.use_raw})")

            try:
                next_state, next_player_id = env.step(step_action, ai_agent.use_raw)
                logger.debug(f"✅ [AI_TURN] env.step() successful, next_player: {next_player_id}")
            except Exception as step_error:
                logger.error(f"❌ [AI_TURN] env.step() failed: {step_error}")
                logger.error(traceback.format_exc())
                return None

            # Edge case: Environment error
            if next_state is None:
                logger.error(f"❌ [AI_TURN] Environment returned None state for AI turn in session {session_id}")
                return None
            
            # REMOVED: BB-first action order override logic (70+ lines)
            # This functionality is now handled natively by the modified RLCard fork
            # The fork implements BB-first postflop action order directly in the game engine

            # Update game state - use RLCard's natural next_player_id
            game['current_state'] = next_state
            game['current_player'] = next_player_id
            
            # Get updated state info for logging
            next_raw_obs = next_state.get('raw_obs', {}) if next_state else {}
            next_pot = next_raw_obs.get('pot', 0)
            next_public_cards = next_raw_obs.get('public_cards', [])
            next_raised = next_raw_obs.get('raised', [0, 0])
            
            # Log AI action result
            logger.info(f"🔄 AI action processed - Session: {session_id}, Next: P{next_player_id}, Pot: {next_pot}, Cards: {len(next_public_cards)}")

            # Track decision for hand history
            logger.debug(f"📊 [AI_TURN] Tracking decision - Player: {current_player}, Action: {action}")

            # Update state's current_player to reflect the player who just acted
            state['raw_obs']['current_player'] = current_player

            # Calculate bet amount for event emission
            bet_amount = None
            try:
                # Get raised amounts before and after action
                previous_raised = raw_obs.get('raised', [0, 0])
                next_raised = next_raw_obs.get('raised', [0, 0])
                
                if action_value in [2, 3, 4]:  # Raise actions
                    # For raises, bet_amount is the total amount raised TO
                    if current_player == 0:
                        bet_amount = next_raised[0] if len(next_raised) > 0 else 0
                    else:
                        bet_amount = next_raised[1] if len(next_raised) > 1 else 0
                elif action_value == 1:  # Check/Call
                    # For Check/Call, calculate the amount needed to call using PREVIOUS state
                    # After a call, raised array equalizes, so we need to use before-state
                    if current_player == 0:
                        opponent_raised_before = previous_raised[1] if len(previous_raised) > 1 else 0
                        player_raised_before = previous_raised[0] if len(previous_raised) > 0 else 0
                        call_amount = opponent_raised_before - player_raised_before
                        bet_amount = call_amount if call_amount > 0 else 0
                    else:
                        opponent_raised_before = previous_raised[0] if len(previous_raised) > 0 else 0
                        player_raised_before = previous_raised[1] if len(previous_raised) > 1 else 0
                        call_amount = opponent_raised_before - player_raised_before
                        bet_amount = call_amount if call_amount > 0 else 0
            except Exception as e:
                logger.warning(f"Error calculating bet_amount for AI event: {e}")
                bet_amount = None

            # Emit action event
            self._emit_action_event(session_id, current_player, action_value, state, next_state, bet_amount)
            
            # Also track decision for legacy compatibility
            self._track_decision(session_id, current_player, action, state, next_state)

            return self.get_game_state(session_id)
        except Exception as e:
            logger.error(f"Error processing AI turn for session {session_id}: {str(e)}")
            # Return current state on error to prevent game from breaking
            return self.get_game_state(session_id)
    
    def _track_decision(self, session_id, player_id, action, state, next_state):
        """Track a decision for hand history - SIMPLIFIED for native RLCard integration"""
        logger.info(f"🔍 [TRACK_DECISION] Starting - session_id={session_id}, player_id={player_id}, action={action}")
        try:
            logger.info(f"🔍 [TRACK_DECISION] Step 1: Checking hand_history_storage...")
            if session_id not in hand_history_storage:
                hand_history_storage[session_id] = []
                logger.info(f"🔍 [TRACK_DECISION] Step 1: Created new hand_history_storage entry")

            # SIMPLIFIED: Use native RLCard state directly (no complex reconstruction needed)
            # Use next_state for pot and public cards (after the action), but state for other context
            logger.info(f"🔍 [TRACK_DECISION] Step 2: Extracting raw_obs from state and next_state...")
            raw_obs = state.get('raw_obs', {})
            next_raw_obs = next_state.get('raw_obs', {}) if next_state else {}
            logger.info(f"🔍 [TRACK_DECISION] Step 2: raw_obs keys: {list(raw_obs.keys()) if raw_obs else 'empty'}, next_raw_obs keys: {list(next_raw_obs.keys()) if next_raw_obs else 'empty'}")

            # Get stage directly from RLCard state (now consistent)
            logger.info(f"🔍 [TRACK_DECISION] Step 3: Getting stage...")
            stage = raw_obs.get('stage', 0)
            logger.info(f"🔍 [TRACK_DECISION] Step 3a: stage (raw)={stage} (type: {type(stage)})")
            if hasattr(stage, 'value'):
                stage = stage.value
                logger.info(f"🔍 [TRACK_DECISION] Step 3b: stage.value={stage}")
            elif not isinstance(stage, int):
                stage = safe_int_convert(stage, 0)
                logger.info(f"🔍 [TRACK_DECISION] Step 3c: converted stage={stage}")
            logger.info(f"🔍 [TRACK_DECISION] Step 3: Final stage={stage} (type: {type(stage)})")

            # SIMPLIFIED: Get hand directly from RLCard's consistent state
            # With native BB-first action order, state should be consistent
            logger.info(f"🔍 [TRACK_DECISION] Step 4: Extracting hand cards...")
            hand_cards = []
            if session_id in self.games:
                game = self.games[session_id]
                env = game.get('env')
                logger.info(f"🔍 [TRACK_DECISION] Step 4a: session_id in games: {session_id in self.games}, env exists: {env is not None}")
                if env and hasattr(env, 'game') and hasattr(env.game, 'players') and player_id < len(env.game.players):
                    logger.info(f"🔍 [TRACK_DECISION] Step 4b: player_id={player_id}, len(players)={len(env.game.players) if hasattr(env.game, 'players') else 'N/A'}")
                    player_obj = env.game.players[player_id]
                    if hasattr(player_obj, 'hand') and player_obj.hand:
                        logger.info(f"🔍 [TRACK_DECISION] Step 4c: player_obj.hand exists: {player_obj.hand}")
                        # Convert Card objects to strings for JSON serialization
                        try:
                            from rlcard.games.base import Card
                            if isinstance(player_obj.hand[0], Card):
                                hand_cards = [str(card) for card in player_obj.hand]
                            else:
                                hand_cards = [str(c) for c in player_obj.hand]
                            logger.info(f"🔍 [TRACK_DECISION] Step 4d: Converted hand_cards: {hand_cards}")
                        except Exception as e:
                            logger.error(f"❌ [TRACK_DECISION] Step 4d: Error converting hand: {e}")
                            hand_cards = [str(c) for c in player_obj.hand]
                else:
                    logger.warning(f"🔍 [TRACK_DECISION] Step 4: Could not access player hand - env={env is not None}, has game={hasattr(env, 'game') if env else False}, player_id valid={player_id < len(env.game.players) if (env and hasattr(env, 'game') and hasattr(env.game, 'players')) else False}")
            logger.info(f"🔍 [TRACK_DECISION] Step 4: Final hand_cards: {hand_cards}")

            # Convert action to serializable format
            logger.info(f"🔍 [TRACK_DECISION] Step 5: Converting action to serializable format...")
            logger.info(f"🔍 [TRACK_DECISION] Step 5a: action={action} (type: {type(action)})")
            if hasattr(action, 'value'):
                action_value = action.value
                logger.info(f"🔍 [TRACK_DECISION] Step 5b: action.value={action_value}")
            else:
                action_value = safe_int_convert(action, 0)
                logger.info(f"🔍 [TRACK_DECISION] Step 5c: converted action_value={action_value}")
            logger.info(f"🔍 [TRACK_DECISION] Step 5: Final action_value={action_value} (type: {type(action_value)})")

            # Get pot from next_state (after the action was taken)
            logger.info(f"🔍 [TRACK_DECISION] Step 5d: Getting pot...")
            pot = next_raw_obs.get('pot', raw_obs.get('pot', 0))
            logger.info(f"🔍 [TRACK_DECISION] Step 5e: pot (raw)={pot} (type: {type(pot)})")
            pot = safe_int_convert(pot, 0)
            logger.info(f"🔍 [TRACK_DECISION] Step 5f: pot (converted)={pot}")

            # Get public cards from next_state (after the action)
            logger.info(f"🔍 [TRACK_DECISION] Step 5g: Getting public cards...")
            public_cards = next_raw_obs.get('public_cards', raw_obs.get('public_cards', []))
            logger.info(f"🔍 [TRACK_DECISION] Step 5h: public_cards={public_cards} (count: {len(public_cards) if public_cards else 0})")

            # Calculate bet amount for this action
            logger.info(f"🔍 [TRACK_DECISION] Step 6: Calculating bet_amount...")
            bet_amount = 0
            try:
                # For action labeling, bet_amount represents the total amount raised TO
                previous_pot = raw_obs.get('pot', 0)
                current_pot = next_raw_obs.get('pot', 0)
                logger.info(f"🔍 [TRACK_DECISION] Step 6a: previous_pot (raw)={previous_pot} (type: {type(previous_pot)}), current_pot (raw)={current_pot} (type: {type(current_pot)})")
                # Convert to int to avoid array comparison errors
                previous_pot = safe_int_convert(previous_pot, 0)
                current_pot = safe_int_convert(current_pot, 0)
                logger.info(f"🔍 [TRACK_DECISION] Step 6b: previous_pot (converted)={previous_pot}, current_pot (converted)={current_pot}")
                pot_increase = current_pot - previous_pot
                logger.info(f"🔍 [TRACK_DECISION] Step 6c: pot_increase={pot_increase}, action_value={action_value}")

                try:
                    pot_increase_check = pot_increase > 0
                    logger.info(f"🔍 [TRACK_DECISION] Step 6d: pot_increase > 0: {pot_increase_check} (type: {type(pot_increase_check)})")
                except Exception as e:
                    logger.error(f"❌ [TRACK_DECISION] Step 6d: Error comparing pot_increase > 0: {e}, pot_increase={pot_increase}, type={type(pot_increase)}")
                    raise

                if pot_increase > 0 and action_value in [2, 3, 4]:  # Raise actions
                    # For raise actions, bet_amount is the amount the player raised TO
                    # This is the new pot value for single-raise actions
                    bet_amount = current_pot
                elif action_value == 1:  # Check/Call
                    # For call actions, bet_amount is the amount called to
                    # This will be determined by ActionLabeling based on opponent_raised
                    bet_amount = 0
                else:
                    bet_amount = 0

            except Exception as e:
                logger.error(f"❌ [TRACK_DECISION] Step 6: Error calculating bet_amount: {e}")
                logger.error(traceback.format_exc())
                bet_amount = 0
            logger.info(f"🔍 [TRACK_DECISION] Step 6: Final bet_amount={bet_amount}")

            # CRITICAL FIX: Store context information for correct action labeling
            # This fixes the bug where Call actions show as "Check" in action history
            logger.info(f"🔍 [TRACK_DECISION] Step 7: Extracting context for action labeling...")
            context = None
            try:
                if session_id in self.games:
                    game = self.games[session_id]
                    env = game.get('env')
                    # Build state dict with both raw_obs and processed fields for ActionLabeling
                    state_with_processed = {
                        'raw_obs': raw_obs,
                        'in_chips': state.get('in_chips', [0, 0]) if isinstance(state, dict) else [0, 0],
                        'raised': raw_obs.get('raised', [0, 0])
                    }
                    logger.info(f"🔍 [TRACK_DECISION] Step 7a: Calling ActionLabeling.get_context_from_state...")
                    context = ActionLabeling.get_context_from_state(state_with_processed, player_id=player_id, env=env)
                    logger.info(f"🔍 [TRACK_DECISION] Step 7b: Got context: {context}")
                    # Convert numpy types to Python native types to prevent boolean evaluation errors
                    if context:
                        logger.info(f"🔍 [TRACK_DECISION] Step 7c: Converting numpy types in context...")
                        context = self._convert_numpy_types_in_dict(context)
                        logger.info(f"🔍 [TRACK_DECISION] Step 7d: Converted context: {context}")
            except Exception as e:
                logger.error(f"❌ [TRACK_DECISION] Step 7: Could not extract context for action labeling: {e}")
                logger.error(traceback.format_exc())
                context = None

            # Create decision record
            logger.info(f"🔍 [TRACK_DECISION] Step 8: Creating decision record...")
            logger.info(f"🔍 [TRACK_DECISION] Step 8a: player_id={player_id}, action_value={action_value}, stage={stage}, pot={pot}")
            logger.info(f"🔍 [TRACK_DECISION] Step 8b: hand_cards={hand_cards}, public_cards={public_cards}, bet_amount={bet_amount}")
            
            decision_record = {
                'player_id': player_id,
                'action': action_value,
                'stage': stage,
                'pot': pot,
                'hand': hand_cards,
                'public_cards': public_cards,
                'bet_amount': bet_amount,
                'context': context,  # Store context for correct action labeling
                'timestamp': time.time()
            }
            logger.info(f"🔍 [TRACK_DECISION] Step 8c: Decision record created: {decision_record}")

            # Add to hand history
            logger.info(f"🔍 [TRACK_DECISION] Step 9: Adding decision record to hand_history_storage...")
            hand_history_storage[session_id].append(decision_record)
            logger.info(f"🔍 [TRACK_DECISION] Step 9: ✓ Decision record added. Total decisions: {len(hand_history_storage[session_id])}")

            # Log the decision (simplified)
            stage_names = {0: 'Preflop', 1: 'Flop', 2: 'Turn', 3: 'River'}
            stage_name = stage_names.get(stage, f'Stage {stage}')
            action_name = f'Action {action_value}'
            hand_str = ', '.join(hand_cards) if hand_cards else 'None'
            public_cards_str = ', '.join([str(c) for c in public_cards]) if public_cards else 'None'

            logger.info(f"📊 [TRACK_DECISION] Decision tracked - Session: {session_id}, Player {player_id}, {stage_name}, Action: {action_value}, Pot: {pot}, Hand: [{hand_str}], Board: [{public_cards_str}]")
            logger.info(f"🔍 [TRACK_DECISION] Step 10: ✓ _track_decision completed successfully")

        except Exception as e:
            logger.error(f"❌ [TRACK_DECISION] ERROR in _track_decision for session {session_id}: {str(e)}")
            logger.error(f"❌ [TRACK_DECISION] Error type: {type(e).__name__}")
            logger.error(f"❌ [TRACK_DECISION] Full traceback:")
            logger.error(traceback.format_exc())
            # Don't fail the game if tracking fails

    def _convert_numpy_types_in_dict(self, data):
        """
        Recursively convert numpy types in a dictionary to Python native types.
        This prevents numpy boolean evaluation errors when processing hand history data.
        """
        if not isinstance(data, dict):
            return data

        converted = {}
        for key, value in data.items():
            try:
                # Import numpy locally to avoid import issues
                import numpy as np
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    converted[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    converted[key] = float(value)
                elif isinstance(value, np.bool_):
                    converted[key] = bool(value)
                elif isinstance(value, np.ndarray):
                    converted[key] = value.tolist()
                elif isinstance(value, dict):
                    converted[key] = self._convert_numpy_types_in_dict(value)
                elif isinstance(value, (list, tuple)):
                    converted[key] = [self._convert_numpy_types_in_dict(item) if isinstance(item, dict) else item for item in value]
                else:
                    converted[key] = value
            except (NameError, AttributeError, ImportError):
                # numpy not available, just copy the value
                converted[key] = value

        return converted

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
            logger.error(f"Game creation failed for session {session_id} - start_game returned None")
            return jsonify({'error': 'Failed to start game'}), 500

        return jsonify(game_state), 200

    except Exception as e:
        logger.error(f"Error starting game: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


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
            logger.error(f"❌ [API] Invalid JSON request")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        session_id = data.get('session_id')
        action_value = data.get('action_value')

        logger.info(f"🎯 [API] Action request - {session_id}: action={action_value}")

        if session_id is None:
            return jsonify({'error': 'Missing required field: session_id'}), 400

        if action_value is None:
            return jsonify({'error': 'Missing required field: action_value'}), 400
        
        # Validate action before processing
        if session_id in game_manager.games:
            processed_state = game_manager.get_game_state(session_id)
            if processed_state:
                legal_actions = processed_state.get('raw_legal_actions', [])
                raw_state = game_manager.games[session_id]['current_state']

                is_valid, error_msg = ActionValidator.validate_action(
                    action_value, raw_state, player_id=0, legal_actions=legal_actions
                )

                if not is_valid:
                    logger.warning(f"❌ [API] Invalid action {action_value} for {session_id}: {error_msg}")
                    return jsonify({'error': error_msg}), 400
        
        # Check if session exists before processing
        if session_id not in game_manager.games:
            logger.warning(f"Game session not found for action: {session_id}")
            return jsonify({
                'error': 'Game session not found. Please start a new game.',
                'session_id': session_id
            }), 404

        # Get game and check if it's over first
        game = game_manager.games[session_id]
        env = game.get('env')
        
        # Check if game is over before checking turn
        if env:
            try:
                is_over = safe_bool_convert(env.is_over())
                if is_over:
                    logger.warning(f"❌ [API] Game already over: {session_id}")
                    # Return current game state instead of error
                    game_state = game_manager.get_game_state(session_id)
                    return jsonify(game_state), 200
            except Exception as e:
                logger.error(f"❌ [API] Error checking if game is over: {e}")

        # Check if it's the human player's turn (player 0)
        current_player = game['current_player']
        if current_player != 0:
            logger.warning(f"❌ [API] Not player's turn: {session_id} (current: P{current_player})")
            return jsonify({
                'error': 'Not your turn to act',
                'current_player': current_player,
                'waiting_for': 'opponent' if current_player == 1 else 'unknown'
            }), 400

        # Process action
        game_state = game_manager.process_action(session_id, action_value)
        
        if game_state is None:
            logger.error(f"process_action returned None for session {session_id}, action_value {action_value}")
            return jsonify({
                'error': 'Failed to process action. The game state may be invalid.',
                'session_id': session_id,
                'action_value': action_value
            }), 500
        
        return jsonify(game_state), 200
    
    except ValueError as e:
        logger.error(f"❌ [API] ValueError in action processing: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"❌ [API] Unexpected error processing action: {str(e)}")
        logger.error(f"❌ [API] Full traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/api/game/button-labels', methods=['POST'])
def get_button_labels():
    """Get button labels for current game context"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id is None:
            return jsonify({'error': 'Missing required field: session_id'}), 400
        
        # Check if session exists
        if session_id not in game_manager.games:
            return jsonify({
                'error': 'Game session not found. Please start a new game.',
                'session_id': session_id
            }), 404
        
        # Get game state
        game_state = game_manager.get_game_state(session_id)
        if game_state is None:
            return jsonify({'error': 'Failed to get game state'}), 500
        
        # Extract context using ActionLabeling
        game = game_manager.games[session_id]
        env = game['env']
        raw_state = game['current_state']
        player_id = 0  # Human player

        # Build state dict with both raw_obs and processed fields (in_chips, raised)
        # game_state has top-level fields from get_game_state, which are more reliable
        state_with_processed = {
            'raw_obs': raw_state.get('raw_obs', {}),
            'in_chips': game_state.get('in_chips', [0, 0]),
            'raised': game_state.get('raised', [0, 0])
        }

        if 'raw_obs' in state_with_processed:
            raw_obs = state_with_processed['raw_obs']

        context = ActionLabeling.get_context_from_state(state_with_processed, player_id=player_id, env=env)
        labels = ActionLabeling.get_button_labels(context)

        return jsonify(labels), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error getting button labels: {str(e)}")
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
            logger.info(f"[HAND_HISTORY_DEBUG] Raw hand_history from storage: {hand_history}")
            if hand_history:
                # Log sample of hand history structure for debugging
                sample = hand_history[-1] if len(hand_history) > 0 else {}
                logger.debug(f"Sample decision structure: {sample}")
                player_decisions = [d for d in hand_history if d.get('player_id') == 0 or d.get('player_id') == "0" or str(d.get('player_id', '')) == "0"]
                logger.info(f"Found {len(player_decisions)} player decisions (player_id == 0) in hand history")
                logger.info(f"[HAND_HISTORY_DEBUG] Player decisions sample: {player_decisions[:2] if player_decisions else 'None'}")

                # Debug logging for hand cards in recent decisions
                for i, decision in enumerate(hand_history[-5:], 1):  # Last 5 decisions
                    hand_cards = decision.get('hand', [])
                    player_id = decision.get('player_id')
                    if hand_cards:
                        logger.info(f"[CHAT_DEBUG] Decision {len(hand_history)-5+i}: player_id={player_id}, hand_cards={hand_cards}")

            # Comprehensive logging for chat request
            logger.info(f"[CHAT_REQUEST] Message: '{message}'")
            logger.info(f"[CHAT_REQUEST] Session: {session_id}")
            logger.info(f"[CHAT_REQUEST] Game context: {game_context is not None}")
            logger.info(f"[CHAT_REQUEST] Hand history: {len(hand_history) if hand_history else 0} decisions")

            if hand_history:
                player_decisions = [d for d in hand_history if d.get('player_id') == 0 or d.get('player_id') == "0" or str(d.get('player_id', '')) == "0"]
                logger.info(f"[CHAT_REQUEST] Player decisions: {len(player_decisions)}")
                if player_decisions:
                    first_player_decision = player_decisions[0]
                    logger.info(f"[CHAT_REQUEST] First player decision hand: {first_player_decision.get('hand', [])}")

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
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')
    port = int(os.getenv('PORT', '5002'))  # Use 5002 as default instead of 5001
    # Use 0.0.0.0 for production deployments (like Render) to bind to all interfaces
    host = '0.0.0.0' if os.getenv('RENDER') else '127.0.0.1'
    logger.info(f"🚀 Starting Flask app on http://{host}:{port} (debug={debug_mode})")
    app.run(host=host, port=port, debug=debug_mode)
