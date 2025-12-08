"""
LLM-Based Opponent Agent for No Limit Hold'em

This agent uses LLM (OpenAI/OpenRouter) with tool calling to make GTO-optimal decisions
instead of rules-based logic. The LLM receives complete game context and selects
actions using tool calling.

Replaces GTOAgent functionality with LLM-powered decision making.
"""

import os
import time
import logging
import traceback
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception as e:
    OPENAI_AVAILABLE = False
    OpenAI = None
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Configure logging
logger = logging.getLogger(__name__)

# Import shared action labeling module
from .action_labeling import ActionLabeling

# Import RLCard Action enum
try:
    from rlcard.games.nolimitholdem.round import Action
except ImportError:
    # Fallback if import fails
    class Action:
        FOLD = 0
        CHECK_CALL = 1
        RAISE_HALF_POT = 2
        RAISE_POT = 3
        ALL_IN = 4

# Load environment variables
try:
    load_dotenv()
except (PermissionError, FileNotFoundError):
    # .env file not accessible, continue with system environment variables
    pass

# Configure logging


class LLMOpponentAgent:
    """
    LLM-based agent that uses OpenAI/OpenRouter with tool calling to make GTO-optimal decisions.
    
    This agent:
    - Uses LLM to make all decisions (preflop and postflop)
    - Receives comprehensive game context (cards, positions, action history, stacks, pot odds, equity)
    - Uses tool calling to select from legal actions
    - Falls back to GTOAgent on LLM failures
    """
    
    def __init__(self, num_actions):
        """
        Initialize LLM opponent agent.
        
        Args:
            num_actions (int): Number of available actions
        """
        self.use_raw = True  # Use raw observations
        self.num_actions = num_actions
        
        # API timeout: 12 seconds (matches ChatbotCoach)
        self.api_timeout = 12.0
        # Executor timeout: 15 seconds (maximum time for entire LLM call including retries)
        self.executor_timeout = 15.0
        
        # Initialize LLM client (reuse ChatbotCoach pattern)
        self._init_llm_client()
        
        # Initialize GTOAgent for fallback
        from .gto_agent import GTOAgent
        self.gto_agent = GTOAgent(num_actions)
        
        # Initialize EquityCalculator and StrategyEvaluator for pre-calculated analysis
        from .equity_calculator import EquityCalculator
        from .strategy_evaluator import StrategyEvaluator
        self.equity_calculator = EquityCalculator()
        self.strategy_evaluator = StrategyEvaluator()
        
        # Thread pool executor for non-blocking API calls
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Load card2index mapping for converting card strings to indices
        try:
            import rlcard
            with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
                self.card2index = json.load(file)
        except Exception as e:
            # Fallback card2index mapping for basic poker cards
            # Format: [rank][suit] where rank: 2-9,T,J,Q,K,A and suit: S,H,D,C
            self.card2index = {}
            suits = ['S', 'H', 'D', 'C']
            ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
            card_index = 0
            for rank in ranks:
                for suit in suits:
                    self.card2index[f"{rank}{suit}"] = card_index
                    card_index += 1
    
    def _init_llm_client(self):
        """
        Initialize LLM client using LLM_PROVIDER environment variable.
        Defaults to OpenAI if not specified. Reuses ChatbotCoach pattern.
        """
        # Check LLM_PROVIDER environment variable
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

        # Get API keys
        openrouter_key = os.getenv("OPEN_ROUTER_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        self.use_openrouter = False
        self.model = "gpt-4-turbo-preview"  # Default model
        self.api_key_available = False
        self.max_history_actions = 3  # Limit action history for token savings
        self.llm_provider = llm_provider

        # Validate provider choice and initialize appropriate client
        if llm_provider == "openrouter":
            if not openrouter_key or openrouter_key == "your_openrouter_key_here" or openrouter_key == "your_key_here":
                self.client = None
                self.api_key_available = False
                logger.warning(f"🤖 [LLM_OPPONENT] OpenRouter key not available or invalid")
            else:
                try:
                    # OpenRouter uses OpenAI-compatible API with different base URL
                    self.client = OpenAI(
                        api_key=openrouter_key,
                        base_url="https://openrouter.ai/api/v1",
                        timeout=self.api_timeout
                    )
                    self.api_key_available = True
                    self.use_openrouter = True
                    # Use a good default model for OpenRouter (GPT-4 Turbo)
                    self.model = "openai/gpt-4-turbo"
                    logger.info(f"🤖 [LLM_OPPONENT] OpenRouter client initialized successfully")
                except Exception as e:
                    self.client = None
                    self.api_key_available = False
                    logger.warning(f"🤖 [LLM_OPPONENT] Failed to initialize OpenRouter client: {e}")

        elif llm_provider == "openai":
            if not OPENAI_AVAILABLE:
                self.client = None
                self.api_key_available = False
                logger.warning(f"🤖 [LLM_OPPONENT] OpenAI library not available")
            elif not openai_key or openai_key == "your_openai_api_key_here" or openai_key == "your_key_here":
                self.client = None
                self.api_key_available = False
                logger.warning(f"🤖 [LLM_OPPONENT] OpenAI key not available or invalid")
            else:
                try:
                    self.client = OpenAI(
                        api_key=openai_key,
                        timeout=self.api_timeout
                    )
                    self.api_key_available = True
                    self.model = "gpt-4-turbo-preview"
                    logger.info(f"🤖 [LLM_OPPONENT] OpenAI client initialized successfully")
                except Exception as e:
                    self.client = None
                    self.api_key_available = False
                    logger.warning(f"🤖 [LLM_OPPONENT] Failed to initialize OpenAI client: {e}")

        else:
            # Try OpenAI as fallback
            if openai_key and openai_key != "your_openai_api_key_here" and openai_key != "your_key_here":
                try:
                    self.client = OpenAI(
                        api_key=openai_key,
                        timeout=self.api_timeout
                    )
                    self.api_key_available = True
                    self.model = "gpt-4-turbo-preview"
                    self.llm_provider = "openai"
                    logger.info(f"🤖 [LLM_OPPONENT] OpenAI client initialized successfully (fallback)")
                except Exception as e:
                    self.client = None
                    self.api_key_available = False
                    logger.warning(f"🤖 [LLM_OPPONENT] Failed to initialize OpenAI client (fallback): {e}")
            else:
                self.client = None
                self.api_key_available = False
                logger.warning(f"🤖 [LLM_OPPONENT] No valid API keys found, LLM opponent will use GTOAgent fallback")
    
    def _card_to_rank_suit(self, card_index):
        """
        Convert RLCard card index to rank and suit.
        
        RLCard uses 0-51: suit = card // 13, rank = card % 13
        Rank: 0=A, 1=2, 2=3, ..., 12=K
        
        Args:
            card_index (int): RLCard card index (0-51)
        
        Returns:
            tuple: (rank, suit) where rank is 0-12 (A=0, K=12) and suit is 0-3
        
        Raises:
            ValueError: If card_index cannot be converted to valid integer or is out of range
        """
        # Convert to int if it's a string or numpy type
        if not isinstance(card_index, int):
            try:
                card_index = int(card_index)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert card_index {card_index} to integer: {e}")
        
        # Validate card index is in valid range (0-51)
        if card_index < 0 or card_index > 51:
            raise ValueError(f"Card index {card_index} is out of valid range (0-51)")
        
        suit = card_index // 13
        rank = card_index % 13
        return (rank, suit)
    
    def _hand_to_string(self, hand):
        """
        Convert RLCard hand (list of card indices) to hand string format.
        
        Args:
            hand (list): List of 2 card indices
        
        Returns:
            str: Hand string like "AA", "AKs", "AKo", "72o"
        """
        if not hand or len(hand) != 2:
            return None
        
        rank_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        
        card1_rank, card1_suit = self._card_to_rank_suit(hand[0])
        card2_rank, card2_suit = self._card_to_rank_suit(hand[1])
        
        rank1_name = rank_names[card1_rank]
        rank2_name = rank_names[card2_rank]
        
        # Ensure rank1 >= rank2 (higher rank first)
        # Note: In poker, Ace (rank 0) is high, so we need special handling
        # For comparison: Ace (0) > King (12) > ... > 2 (1)
        def rank_value(rank):
            # Ace is high, so return 13 for Ace, otherwise return rank + 1
            return 13 if rank == 0 else rank + 1
        
        if rank_value(card1_rank) < rank_value(card2_rank):
            rank1_name, rank2_name = rank2_name, rank1_name
            card1_rank, card2_rank = card2_rank, card1_rank
            card1_suit, card2_suit = card2_suit, card1_suit
        
        # Check if pair
        if card1_rank == card2_rank:
            return f"{rank1_name}{rank2_name}"
        
        # Check if suited
        if card1_suit == card2_suit:
            return f"{rank1_name}{rank2_name}s"
        else:
            return f"{rank1_name}{rank2_name}o"
    
    def _extract_opponent_cards(self, state):
        """
        Extract opponent's hole cards from RLCard game state.
        Opponent is always player 1 (big blind) in heads-up.

        IMPORTANT: In RLCard, when it's player 1's turn, the state passed to the agent
        is from player 1's perspective, so raw_obs['hand'] contains player 1's cards.

        Args:
            state (dict): State dictionary with 'raw_obs'

        Returns:
            list: List of 2 card indices (0-51), or None if not available
        """
        raw_obs = state.get('raw_obs', {})
        hand = raw_obs.get('hand', [])

        if not hand or len(hand) != 2:
            return None

        # Convert cards to integers - handle both string representations (e.g., 'ST', 'DQ')
        # and integer indices (0-51)
        hand_ints = []
        for card in hand:
            try:
                if isinstance(card, str) and self.card2index:
                    # Card is a string like 'ST' - convert using card2index mapping
                    if card in self.card2index:
                        hand_ints.append(self.card2index[card])
                    else:
                        return None
                else:
                    # Card is already an integer or can be converted to int
                    hand_ints.append(int(card))
            except (ValueError, TypeError, KeyError) as e:
                return None

        return hand_ints
    
    def _normalize_card_string(self, card_str):
        """
        Normalize card string to consistent format (Rank+Suit, e.g., "3h", "Ah").
        Handles both "Rank+Suit" (e.g., "3h", "Ah") and "Suit+Rank" (e.g., "H3", "HA") formats.
        
        Args:
            card_str (str): Card string in any format
        
        Returns:
            str: Normalized card string in "Rank+Suit" format (e.g., "3h", "Ah")
        """
        if not isinstance(card_str, str) or len(card_str) < 2:
            return card_str.upper() if isinstance(card_str, str) else str(card_str)
        
        card_str = card_str.upper().strip()
        
        # Map suit letters
        suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c', '♠': 's', '♥': 'h', '♦': 'd', '♣': 'c'}
        rank_map = {'A': 'A', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', 
                   '8': '8', '9': '9', 'T': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K'}
        
        # Check if format is "Suit+Rank" (e.g., "H3", "SA")
        if len(card_str) == 2:
            first_char = card_str[0]
            second_char = card_str[1]
            
            # If first char is a suit and second is a rank, convert to "Rank+Suit"
            if first_char in suit_map and second_char in rank_map:
                return f"{rank_map[second_char]}{suit_map[first_char]}"
            # If first char is a rank and second is a suit, already in correct format
            elif first_char in rank_map and second_char in suit_map:
                return f"{rank_map[first_char]}{suit_map[second_char]}"
        
        # If we can't normalize, return uppercase version
        return card_str
    
    def _card_index_to_string(self, card_index):
        """
        Convert RLCard card index to string format (e.g., "Ah", "Kd").
        
        Args:
            card_index (int): RLCard card index (0-51)
        
        Returns:
            str: Card string like "Ah", "Kd", "2c", "Ts"
        
        Raises:
            ValueError: If card_index is invalid or out of range
        """
        try:
            rank, suit = self._card_to_rank_suit(card_index)
            rank_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
            suit_names = ['c', 'd', 'h', 's']  # clubs, diamonds, hearts, spades
            
            if rank < 0 or rank >= len(rank_names) or suit < 0 or suit >= len(suit_names):
                raise ValueError(f"Invalid rank {rank} or suit {suit} for card_index {card_index}")
            
            return f"{rank_names[rank]}{suit_names[suit]}"
        except (ValueError, IndexError) as e:
            raise
    
    def step(self, state):
        """
        Predict action given current state using LLM.
        
        Args:
            state (dict): State dictionary with 'raw_obs' and 'raw_legal_actions'
        
        Returns:
            action: Action enum value (Action.FOLD, Action.CHECK_CALL, etc.)
        """
        raw_legal_actions = state.get('raw_legal_actions', [])

        if not raw_legal_actions:
            # No legal actions - shouldn't happen
            return Action.FOLD

        # Check if LLM is available
        if not self.api_key_available or not self.client:
            logger.warning(f"🤖 [LLM_OPPONENT] LLM not available - api_key_available={self.api_key_available}, client={self.client is not None}, falling back to GTOAgent")
            logger.warning(f"🤖 [LLM_OPPONENT] LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'not set')}, OPEN_ROUTER_KEY={'set' if os.getenv('OPEN_ROUTER_KEY') else 'not set'}, OPENAI_API_KEY={'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
            return self.gto_agent.step(state)

        # SPECIAL CASE: SB opening preflop - skip LLM and use GTO strategy directly
        raw_obs = state.get('raw_obs', {})
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        elif not isinstance(stage, int):
            stage = int(stage) if stage else 0

        # Check if this is a preflop SB opening scenario
        if stage == 0:  # Preflop
            # Determine if opponent is SB
            dealer_id = state.get('dealer_id')
            opponent_is_sb = False
            if dealer_id is not None:
                opponent_is_sb = (dealer_id == 0)  # If user is dealer (player 0), opponent (player 1) is SB
            else:
                # Fallback: assume opponent is SB if no dealer_id (legacy behavior)
                opponent_is_sb = True

            # Check if no one has raised yet (opening scenario)
            raised = raw_obs.get('raised', [0, 0])
            big_blind = raw_obs.get('big_blind', 2)
            pot = raw_obs.get('pot', 0)
            pot_bb = pot / big_blind if big_blind > 0 else 0

            # Opening scenario: pot is just blinds (~1.5BB) and no raises
            is_opening = (pot_bb <= 2.0 and
                         (len(raised) < 2 or raised[1] <= big_blind * 1.1))

            if opponent_is_sb and is_opening:
                return self.gto_agent.step(state)

            # SPECIAL CASE: BB facing 3BB raise preflop - skip LLM and use GTO strategy directly
            # Determine if opponent is BB
            dealer_id = state.get('dealer_id')
            opponent_is_bb = False
            if dealer_id is not None:
                opponent_is_bb = (dealer_id == 1)  # If opponent is dealer (player 1), opponent is BB
            else:
                # Fallback: assume opponent is BB if no dealer_id (legacy behavior)
                opponent_is_bb = True

            # Check if facing a 3BB raise (button open)
            raised = raw_obs.get('raised', [0, 0])
            big_blind = raw_obs.get('big_blind', 2)
            pot = raw_obs.get('pot', 0)
            pot_bb = pot / big_blind if big_blind > 0 else 0
            
            # User is player 0, opponent is player 1
            user_raised = raised[0] if len(raised) > 0 else 0
            opponent_raised = raised[1] if len(raised) > 1 else 0
            
            # Detect 3BB raise: user raised to ~3BB total (2.5-3.5BB range)
            user_raised_bb = user_raised / big_blind if big_blind > 0 else 0
            facing_3bb_raise = (
                opponent_is_bb and
                user_raised_bb >= 2.5 and user_raised_bb <= 3.5 and  # User raised to ~3BB
                opponent_raised <= big_blind * 1.1  # Opponent hasn't raised yet (just BB)
            )
            
            # Alternative detection using pot size (more reliable when raised array is wrong)
            # Pot after 3BB open = SB (0.5BB) + BB (1BB) + raise (2BB) = 3.5BB
            pot_indicates_3bb_raise = (
                opponent_is_bb and
                pot_bb >= 3.0 and pot_bb <= 4.5 and  # Pot indicates 3BB open
                opponent_raised <= big_blind * 1.1  # Opponent hasn't raised yet
            )
            
            if facing_3bb_raise or pot_indicates_3bb_raise:
                return self.gto_agent.step(state)

        try:
            # 1. Extract opponent cards
            opponent_cards = self._extract_opponent_cards(state)
            if not opponent_cards:
                logger.warning(f"🤖 [LLM_OPPONENT] Could not extract opponent cards, falling back to GTOAgent. State keys: {list(state.keys())}, raw_obs keys: {list(state.get('raw_obs', {}).keys())}")
                return self.gto_agent.step(state)
            
            # 2. Build context
            context = self._build_context(state)
            
            
            # 4. Format context for prompt
            formatted_context = self._format_context_for_prompt(context)
            
            # 5. Call LLM with tool calling
            llm_action = self._call_llm_for_decision(context)
            
            if not llm_action:
                logger.warning(f"🤖 [LLM_OPPONENT] LLM call returned None, falling back to GTOAgent")
                return self.gto_agent.step(state)
            
            # 5. Map LLM action to RLCard Action enum and validate
            legal_actions_labels = context.get('legal_actions_labels', {})
            action = self._map_llm_action_to_rlcard(llm_action, raw_legal_actions, legal_actions_labels=legal_actions_labels)
            
            # 6. Check if action is illegal (shouldn't happen after mapping, but double-check)
            if action not in raw_legal_actions:
                # Retry with clarification about legal actions
                clarified_context = context.copy()
                clarified_context['clarification'] = f"Previous action '{llm_action}' was not legal. Legal actions are: {[context['legal_actions_labels'].get(a, f'Action {a}') for a in raw_legal_actions]}"
                
                retry_action = self._call_llm_for_decision(clarified_context)
                if retry_action:
                    action = self._map_llm_action_to_rlcard(retry_action, raw_legal_actions)
                    if action not in raw_legal_actions:
                        return self.gto_agent.step(state)
                else:
                    return self.gto_agent.step(state)
            
            return action
            
        except Exception as e:
            # Fallback to GTOAgent on any error
            logger.error(f"🤖 [LLM_OPPONENT] Exception in LLM opponent step: {e}")
            logger.error(traceback.format_exc())
            return self.gto_agent.step(state)
    
    def eval_step(self, state):
        """
        Evaluation step - same as step but returns action and info tuple.
        
        Args:
            state (dict): State dictionary
        
        Returns:
            tuple: (action, info) where info contains action probabilities
        """
        action = self.step(state)
        
        # Create probability distribution (all probability on chosen action)
        raw_legal_actions = state.get('raw_legal_actions', [])
        probs = [0.0] * len(raw_legal_actions)
        
        if action in raw_legal_actions:
            action_idx = raw_legal_actions.index(action)
            probs[action_idx] = 1.0
        
        info = {
            'probs': {raw_legal_actions[i]: probs[i] for i in range(len(raw_legal_actions))}
        }
        
        return action, info
    
    def _build_action_history(self, state):
        """
        Build action history from game state.
        Uses actual action history if available, otherwise infers from raised amounts.
        
        Args:
            state (dict): State dictionary with 'raw_obs' and optionally 'action_history_with_cards'
        
        Returns:
            list: List of action history entries
        """
        raw_obs = state.get('raw_obs', {})
        
        # FIRST: Try to use actual action history if available (preferred method)
        if 'action_history_with_cards' in state:
            action_history_with_cards = state['action_history_with_cards']
            
            # Convert game manager's action history format to LLM agent's expected format
            converted_history = []
            big_blind = raw_obs.get('big_blind', 2)
            stakes = raw_obs.get('stakes', [100, 100])
            
            # Determine positions based on dealer_id
            dealer_id = state.get('dealer_id')
            if dealer_id is not None:
                user_is_dealer = (dealer_id == 0)
                if user_is_dealer:
                    user_position = 'big_blind'
                    opponent_position = 'button'
                else:
                    user_position = 'button'
                    opponent_position = 'big_blind'
            else:
                user_position = 'button'
                opponent_position = 'big_blind'
            
            for entry in action_history_with_cards:
                entry_type = entry.get('type', '')
                player_id = entry.get('player_id', -1)
                player_name = entry.get('player_name', '')
                
                # Map player_id to 'user' or 'opponent'
                if player_id == 0:
                    player = 'user'
                    position = user_position
                elif player_id == 1:
                    player = 'opponent'
                    position = opponent_position
                else:
                    continue  # Skip invalid entries
                
                # Convert blind entries
                if entry_type == 'blind':
                    blind_type = entry.get('blind_type', '')
                    amount = entry.get('amount', 0)
                    amount_bb = amount / big_blind if big_blind > 0 else 0
                    
                    converted_history.append({
                        'player': player,
                        'position': 'button' if blind_type == 'small' else 'big_blind',
                        'action': 'blind',
                        'bet_size_chips': amount,
                        'bet_size_bb': amount_bb,
                        'pot_before': 0 if blind_type == 'small' else big_blind // 2,
                        'pot_after': big_blind + (big_blind // 2) if blind_type == 'big' else big_blind // 2,
                        'stack_before': {'user': stakes[0] + amount if player == 'user' else stakes[0],
                                        'opponent': stakes[1] + amount if player == 'opponent' else stakes[1]},
                        'stack_after': {'user': stakes[0] if player == 'user' else stakes[0],
                                       'opponent': stakes[1] if player == 'opponent' else stakes[1]},
                        'stage': 'preflop',
                        'action_label': f'{blind_type.capitalize()} Blind {amount} chips'
                    })
                
                # Convert action entries
                elif entry_type == 'action':
                    action_name = entry.get('action', '')
                    bet_amount = entry.get('bet_amount', 0)
                    
                    # Determine stage from action name or use current stage
                    stage_name = 'preflop'  # Default
                    if 'Flop' in action_name or 'flop' in action_name.lower():
                        stage_name = 'flop'
                    elif 'Turn' in action_name or 'turn' in action_name.lower():
                        stage_name = 'turn'
                    elif 'River' in action_name or 'river' in action_name.lower():
                        stage_name = 'river'
                    
                    # Determine action type from action name
                    action_type = 'unknown'
                    if 'Fold' in action_name or 'fold' in action_name.lower():
                        action_type = 'fold'
                    elif 'Check' in action_name or 'check' in action_name.lower():
                        action_type = 'check'
                    elif 'Call' in action_name or 'call' in action_name.lower():
                        action_type = 'call'
                    elif 'Raise' in action_name or 'raise' in action_name.lower() or 'bet' in action_name.lower():
                        action_type = 'raise'
                    elif 'All-In' in action_name or 'all-in' in action_name.lower():
                        action_type = 'all_in'
                    
                    # Calculate bet size in BB
                    bet_size_bb = bet_amount / big_blind if big_blind > 0 else 0
                    
                    # Get pot size (approximate from current pot)
                    pot = raw_obs.get('pot', 0)
                    
                    converted_history.append({
                        'player': player,
                        'position': position,
                        'action': action_type,
                        'bet_size_chips': bet_amount,
                        'bet_size_bb': bet_size_bb,
                        'pot_before': pot - bet_amount if bet_amount > 0 else pot,
                        'pot_after': pot,
                        'stack_before': {'user': stakes[0] + bet_amount if player == 'user' else stakes[0],
                                        'opponent': stakes[1] + bet_amount if player == 'opponent' else stakes[1]},
                        'stack_after': {'user': stakes[0] if player == 'user' else stakes[0],
                                       'opponent': stakes[1] if player == 'opponent' else stakes[1]},
                        'stage': stage_name,
                        'action_label': action_name
                    })
            
            if converted_history:
                return converted_history
        
        # FALLBACK: Infer action history from raised amounts (original logic)
        action_history = []
        
        # Get current game state info
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        elif not isinstance(stage, int):
            stage = int(stage) if stage else 0
        
        stage_names = ['preflop', 'flop', 'turn', 'river']
        current_stage_name = stage_names[stage] if stage < len(stage_names) else 'preflop'
        
        # Get raised amounts to infer action history
        raised = raw_obs.get('raised', [0, 0])
        pot = raw_obs.get('pot', 0)
        big_blind = raw_obs.get('big_blind', 2)
        # Use 'stakes' for remaining chips (not 'all_chips')
        stakes = raw_obs.get('stakes', [100, 100])  # Default to 100 chips if not available

        
        # Determine positions based on dealer_id (for heads-up poker)
        dealer_id = state.get('dealer_id')
        if dealer_id is not None:
            # In heads-up: dealer is BB, non-dealer is SB (button)
            user_is_dealer = (dealer_id == 0)  # User is player 0
            if user_is_dealer:
                user_position = 'big_blind'
                opponent_position = 'button'  # SB
                sb_player = 'opponent'
                bb_player = 'user'
            else:  # opponent is dealer
                user_position = 'button'  # SB
                opponent_position = 'big_blind'
                sb_player = 'user'
                bb_player = 'opponent'
        else:
            # Fallback to hardcoded positions if dealer_id not available
            sb_player = 'user'
            bb_player = 'opponent'
            user_position = 'button'
            opponent_position = 'big_blind'

        # We can infer some action history from raised amounts and pot size
        # This is a simplified version - full implementation would track actions as they occur

        # Add blinds as first actions
        small_blind = big_blind // 2
        if pot >= small_blind + big_blind:
            action_history.append({
                'player': sb_player,
                'position': 'button',  # SB is always button in heads-up
                'action': 'blind',
                'bet_size_chips': small_blind,
                'bet_size_bb': small_blind / big_blind if big_blind > 0 else 0,
                'pot_before': 0,
                'pot_after': small_blind + big_blind,
                'stack_before': {'user': stakes[0] + small_blind if sb_player == 'user' else stakes[0],
                                'opponent': stakes[1] + small_blind if sb_player == 'opponent' else stakes[1]},
                'stack_after': {'user': stakes[0] if sb_player == 'user' else stakes[0],
                               'opponent': stakes[1] if sb_player == 'opponent' else stakes[1]},
                'stage': 'preflop',
                'action_label': f'Small Blind {small_blind} chips'
            })

            action_history.append({
                'player': bb_player,
                'position': 'big_blind',
                'action': 'blind',
                'bet_size_chips': big_blind,
                'bet_size_bb': 1.0,
                'pot_before': small_blind,
                'pot_after': small_blind + big_blind,
                'stack_before': {'user': stakes[0] + big_blind if bb_player == 'user' else stakes[0],
                                'opponent': stakes[1] + big_blind if bb_player == 'opponent' else stakes[1]},
                'stack_after': {'user': stakes[0] if bb_player == 'user' else stakes[0],
                               'opponent': stakes[1] if bb_player == 'opponent' else stakes[1]},
                'stage': 'preflop',
                'action_label': f'Big Blind {big_blind} chips'
            })
        
        # Infer actions from raised amounts
        # This is simplified - in a full implementation, we'd track actions as they occur
        user_raised = raised[0] if len(raised) > 0 else 0
        opponent_raised = raised[1] if len(raised) > 1 else 0
        
        # Determine if there was a raise by comparing raised amounts
        # If one player has raised more than the other (beyond just the blind), there was a raise
        if stage == 0:  # Preflop
            # Check if user (player 0) raised beyond the big blind
            if user_raised > big_blind:
                bet_size_bb = user_raised / big_blind if big_blind > 0 else 0
                # Detect open raise (2.5-4 BB is typical open raise range)
                if bet_size_bb >= 2.0 and bet_size_bb <= 5.0:
                    action_history.append({
                        'player': 'user',
                        'position': 'button',
                        'action': 'raise',
                        'bet_size_chips': user_raised,
                        'bet_size_bb': bet_size_bb,
                        'pot_before': small_blind + big_blind,
                        'pot_after': pot,
                        'stack_before': {'user': stakes[0] + user_raised if len(stakes) > 0 else 100,
                                        'opponent': stakes[1] if len(stakes) > 1 else 100},
                        'stack_after': {'user': stakes[0] if len(stakes) > 0 else 100,
                                       'opponent': stakes[1] if len(stakes) > 1 else 100},
                        'stage': 'preflop',
                        'action_label': f'Raise to {bet_size_bb:.1f}BB'
                    })
            # Check if opponent (player 1) raised beyond the big blind (3-bet scenario)
            elif opponent_raised > big_blind:
                bet_size_bb = opponent_raised / big_blind if big_blind > 0 else 0
                if bet_size_bb >= 2.0:
                    action_history.append({
                        'player': 'opponent',
                        'position': 'big_blind',
                        'action': 'raise',
                        'bet_size_chips': opponent_raised,
                        'bet_size_bb': bet_size_bb,
                        'pot_before': small_blind + big_blind,
                        'pot_after': pot,
                        'stack_before': {'user': stakes[0] if len(stakes) > 0 else 100,
                                        'opponent': stakes[1] + opponent_raised if len(stakes) > 1 else 100},
                        'stack_after': {'user': stakes[0] if len(stakes) > 0 else 100,
                                       'opponent': stakes[1] if len(stakes) > 1 else 100},
                        'stage': 'preflop',
                        'action_label': f'3-bet to {bet_size_bb:.1f}BB'
                    })
        
        return action_history
    
    def _build_context(self, state):
        """
        Build comprehensive game context dictionary for LLM prompt.
        
        Args:
            state (dict): State dictionary with 'raw_obs' and 'raw_legal_actions'
        
        Returns:
            dict: Context dictionary with all required fields
        """
        raw_obs = state.get('raw_obs', {})
        raw_legal_actions = state.get('raw_legal_actions', [])

        # Store original raised value before any modifications
        original_raised = raw_obs.get('raised', [0, 0])

        # Extract opponent cards
        opponent_cards = self._extract_opponent_cards(state)
        
        
        # Handle both card indices (int) and card strings (str)
        if opponent_cards:
            opponent_cards_str = []
            for card in opponent_cards:
                if isinstance(card, (int, np.integer)):
                    # Card index (0-51) - convert to string
                    opponent_cards_str.append(self._card_index_to_string(card))
                elif isinstance(card, str):
                    # Already a string (e.g., "C9", "ST") - use as-is
                    opponent_cards_str.append(card)
                else:
                    # Try to convert to int first
                    try:
                        card_int = int(card)
                        opponent_cards_str.append(self._card_index_to_string(card_int))
                    except (ValueError, TypeError):
                        opponent_cards_str.append(str(card))
        else:
            opponent_cards_str = []
        
        
        # Get game state info
        stage = raw_obs.get('stage', 0)
        if hasattr(stage, 'value'):
            stage = stage.value
        elif not isinstance(stage, int):
            stage = int(stage) if stage else 0

        stage_names = ['preflop', 'flop', 'turn', 'river']
        current_stage = stage_names[stage] if stage < len(stage_names) else 'preflop'

        
        # Get public cards (board)
        public_cards = raw_obs.get('public_cards', [])
        # Convert cards to strings - handle both integer indices and string formats
        public_cards_str = []
        for card in public_cards:
            try:
                # Handle RLCard Card objects
                try:
                    from rlcard.games.base import Card
                    if isinstance(card, Card):
                        # Convert Card object to index first, then to string
                        card_index = card.get_index()
                        public_cards_str.append(self._card_index_to_string(card_index))
                        continue
                except (ImportError, AttributeError):
                    pass  # Not a Card object, continue with other checks
                
                if isinstance(card, str):
                    # Card is already a string (e.g., "3h", "2h", "2d", "Jd" or "H3", "SA")
                    # Normalize to consistent "Rank+Suit" format
                    normalized = self._normalize_card_string(card)
                    public_cards_str.append(normalized)
                elif isinstance(card, (int, np.integer)):
                    # Card is an integer index (0-51) - convert to string
                    public_cards_str.append(self._card_index_to_string(int(card)))
                else:
                    # Try to convert to int first
                    card_int = int(card)
                    public_cards_str.append(self._card_index_to_string(card_int))
            except (ValueError, TypeError, IndexError) as e:
                # Don't silently default to Ace - log error and skip invalid card
                continue
        
        # Log board cards for debugging
        if public_cards_str:
            pass
        else:
            pass
        
        # Get pot and stack info
        pot = raw_obs.get('pot', 0)
        big_blind = raw_obs.get('big_blind', 2)
        raised = raw_obs.get('raised', [0, 0])

        # CRITICAL FIX: Use 'stakes' for remaining chips (not 'all_chips')
        # In RLCard, 'stakes' represents remained_chips (remaining stack after bets)
        # 'all_chips' may not be populated correctly or may represent something else
        stakes = raw_obs.get('stakes', [100, 100])  # Default to 100 chips if not available

        # Store original raised value to check if it needs calculation
        original_raised = raised[:] if isinstance(raised, list) and len(raised) == 2 else [0, 0]

        # If raised field is missing (which it often is), calculate it from stakes
        # raised represents total bets so far (equivalent to stakes in this context)
        raised_is_default = (isinstance(original_raised, list) and len(original_raised) == 2 and original_raised[0] == 0 and original_raised[1] == 0)
        if raised_is_default:  # raised is default [0,0], so use stakes as raised
            # Use stakes as the raised amounts (total bets so far)
            raised = stakes[:]
        
        pot_size_bb = pot / big_blind if big_blind > 0 else 0
        # Player 0 is user (button), Player 1 is opponent (big blind)
        user_stack = stakes[0] if len(stakes) > 0 else 100
        opponent_stack = stakes[1] if len(stakes) > 1 else 100
        user_stack_bb = user_stack / big_blind if big_blind > 0 else 50
        opponent_stack_bb = opponent_stack / big_blind if big_blind > 0 else 50
        
        # Determine if facing a bet
        # We are player 1 (opponent), user is player 0
        our_raised = raised[1] if len(raised) > 1 else 0  # Our (opponent's) current bet
        opponent_raised = raised[0] if len(raised) > 0 else 0  # User's current bet
        facing_bet = opponent_raised > our_raised
        bet_to_call = max(0, opponent_raised - our_raised) if facing_bet else 0
        bet_to_call_bb = bet_to_call / big_blind if big_blind > 0 else 0
        
        # Build action history (pass corrected raised values)
        # Create a modified state with corrected raised values for action history building
        state_for_history = state.copy()
        if 'raw_obs' in state_for_history:
            state_for_history['raw_obs'] = state_for_history['raw_obs'].copy()
            state_for_history['raw_obs']['raised'] = raised  # Use corrected raised values
        action_history = self._build_action_history(state_for_history)
        
        # Fallback: If facing a bet but no raise in action history, add it
        # This ensures the LLM knows about prior raises even if detection failed
        if facing_bet and current_stage == 'preflop':
            # Check if action history already has a raise from user
            has_user_raise = any(
                action.get('player') == 'user' and action.get('action') == 'raise'
                for action in action_history
            )
            if not has_user_raise and opponent_raised > big_blind:
                bet_size_bb = opponent_raised / big_blind if big_blind > 0 else 0
                # Determine user position
                dealer_id = state.get('dealer_id')
                if dealer_id is not None:
                    user_is_dealer = (dealer_id == 0)
                    user_position = 'big_blind' if user_is_dealer else 'button'
                else:
                    user_position = 'button'
                
                small_blind = big_blind // 2
                action_history.append({
                    'player': 'user',
                    'position': user_position,
                    'action': 'raise',
                    'bet_size_chips': opponent_raised,
                    'bet_size_bb': bet_size_bb,
                    'pot_before': small_blind + big_blind,
                    'pot_after': pot,
                    'stack_before': {'user': user_stack + opponent_raised, 'opponent': opponent_stack},
                    'stack_after': {'user': user_stack, 'opponent': opponent_stack},
                    'stage': 'preflop',
                    'action_label': f'Raise to {bet_size_bb:.1f}BB'
                })
        
        # Calculate pot odds if facing a bet
        pot_odds = 0.0
        if facing_bet and pot > 0:
            pot_odds = bet_to_call / (pot + bet_to_call) if (pot + bet_to_call) > 0 else 0
        
        # Build legal actions labels using shared ActionLabeling module
        # Note: We are player 1 (opponent), so we need to get context from opponent's perspective
        # Create a mock env for context extraction (we'll pass None and extract manually)
        try:
            # Extract context using ActionLabeling (opponent is player 1)
            context = ActionLabeling.get_context_from_state(state, player_id=1, env=None)
            # Override facing_bet with our calculated value (more accurate)
            context['is_facing_bet'] = facing_bet
            button_labels = ActionLabeling.get_button_labels(context)
            
            # Map button labels to action values
            legal_actions_labels = {}
            for action in raw_legal_actions:
                if action == Action.FOLD:
                    legal_actions_labels[action] = 'Fold'
                elif action == Action.CHECK_CALL:
                    legal_actions_labels[action] = button_labels['checkCall']
                elif action == Action.RAISE_HALF_POT:
                    legal_actions_labels[action] = button_labels['raiseHalfPot']
                elif action == Action.RAISE_POT:
                    legal_actions_labels[action] = button_labels['raisePot']
                elif action == Action.ALL_IN:
                    legal_actions_labels[action] = 'All-In'
                else:
                    legal_actions_labels[action] = f'Action {action}'
        except Exception as e:
            # Fallback to original logic if ActionLabeling fails
            legal_actions_labels = {}
            action_label_map = {
                Action.FOLD: 'Fold',
                Action.RAISE_HALF_POT: 'Raise ½ Pot',
                Action.RAISE_POT: 'Raise Pot',
                Action.ALL_IN: 'All-In'
            }
            check_call_label = 'Check' if not facing_bet else 'Call'
            action_label_map[Action.CHECK_CALL] = check_call_label
            for action in raw_legal_actions:
                legal_actions_labels[action] = action_label_map.get(action, f'Action {action}')
        
        # Pre-calculate analysis (optional but recommended for better decisions)
        opponent_range = None
        hand_equity = None
        board_texture = None
        
        try:
            # Convert opponent cards to hand format for range construction
            opponent_hand_str = self._hand_to_string(opponent_cards) if opponent_cards else None
            
            # Construct opponent range (user's likely hands based on actions)
            if action_history:
                # Convert action history to format expected by EquityCalculator
                formatted_action_history = []
                for action_entry in action_history:
                    if action_entry.get('player') == 'user':  # User's actions
                        formatted_action_history.append({
                            'action': action_entry.get('action_label', action_entry.get('action', '')),
                            'stage': 0 if action_entry.get('stage') == 'preflop' else 1,
                            'bet_amount': action_entry.get('bet_size_chips', 0),
                            'pot': action_entry.get('pot_after', 0),
                            'big_blind': big_blind
                        })
                
                if formatted_action_history:
                    opponent_range = self.equity_calculator._construct_opponent_range(
                        formatted_action_history,
                        'button',  # User is button
                        user_stack_bb,
                        current_stage=stage,
                        board=public_cards if stage > 0 else None
                    )
            
            # Calculate hand equity vs opponent range
            if opponent_range and opponent_cards and stage > 0:  # Postflop only
                try:
                    hand_equity = self.equity_calculator.calculate_full_equity(
                        opponent_cards,
                        public_cards,
                        opponent_range,
                        stage
                    )
                except Exception as e:
                    # Fallback to hand strength estimate
                    hand_strength = self.equity_calculator.categorize_hand_strength(
                        opponent_cards, public_cards, stage
                    )
                    hand_equity = self.equity_calculator.estimate_equity(hand_strength.get('category', 'weak'))
            
            # Analyze board texture (postflop only)
            if stage > 0 and public_cards:
                board_texture = self.equity_calculator._analyze_board_texture(public_cards)
        except Exception as e:
            # Continue without pre-calculated values
            pass
        
        # Determine positions based on dealer_id (for heads-up poker)
        dealer_id = state.get('dealer_id')
        if dealer_id is not None:
            # In heads-up: dealer is BB, non-dealer is SB (button)
            user_is_dealer = (dealer_id == 0)  # User is player 0
            opponent_is_dealer = (dealer_id == 1)  # Opponent is player 1

            if user_is_dealer:
                user_position = 'big_blind'
                opponent_position = 'button'  # SB
            else:  # opponent is dealer
                user_position = 'button'  # SB
                opponent_position = 'big_blind'
        else:
            # Fallback to hardcoded positions if dealer_id not available
            user_position = 'button'
            opponent_position = 'big_blind'

        # Build context dictionary
        context = {
            'opponent_cards': opponent_cards_str,
            'user_position': user_position,
            'opponent_position': opponent_position,
            'current_stage': current_stage,
            'public_cards': public_cards_str,
            'pot_size': pot,
            'pot_size_bb': pot_size_bb,
            'big_blind': big_blind,
            'current_stacks': {
                'user': user_stack,
                'opponent': opponent_stack
            },
            'stack_depths': {
                'user': user_stack_bb,
                'opponent': opponent_stack_bb
            },
            'action_history': action_history,
            'legal_actions': raw_legal_actions,
            'legal_actions_labels': legal_actions_labels,
            'facing_bet': facing_bet,
            'bet_to_call': bet_to_call,
            'bet_to_call_bb': bet_to_call_bb,
            'pot_odds': pot_odds,
            # Pre-calculated analysis (optional)
            'opponent_range': list(opponent_range.keys()) if opponent_range else None,
            'hand_equity': hand_equity,
            'board_texture': board_texture
        }
        
        return context
    
    def _format_context_for_prompt(self, context):
        """
        Format context dictionary as concise text block for LLM prompt.

        Args:
            context (dict): Context dictionary

        Returns:
            str: Formatted context text
        """
        lines = []
        lines.append(f"Stage: {context.get('current_stage', 'preflop')}")

        # Cards and board
        opponent_cards = context.get('opponent_cards', [])
        if opponent_cards:
            lines.append(f"Your Cards: {', '.join(opponent_cards)}")
        else:
            lines.append("Your Cards: Hidden")

        public_cards = context.get('public_cards', [])
        if public_cards:
            lines.append(f"Board: {', '.join(public_cards)}")

        # Key numbers
        pot_bb = context.get('pot_size_bb', 0)
        opponent_stack_bb = context.get('stack_depths', {}).get('opponent', 500.0)
        lines.append(f"Pot: {pot_bb:.1f}BB | Your Stack: {opponent_stack_bb:.1f}BB")

        # Current situation
        if context.get('facing_bet', False):
            bet_to_call_bb = context.get('bet_to_call_bb', 0.0)
            pot_odds = context.get('pot_odds', 0.0)
            lines.append(f"Facing bet: {bet_to_call_bb:.1f}BB to call ({pot_odds*100:.1f}% pot odds)")
        else:
            lines.append("First to act")

        # Recent action history (limited to save tokens)
        action_history = context.get('action_history', [])
        if action_history:
            lines.append("Recent Actions:")
            # Only show last N actions to save tokens
            recent_actions = action_history[-self.max_history_actions:]
            for i, action in enumerate(recent_actions, 1):
                stage = action.get('stage', 'preflop')[:1].upper()  # P/F/T/R
                player = action.get('player', 'unknown')[:1].upper()  # O/U
                action_label = action.get('action_label', action.get('action', 'unknown'))
                lines.append(f"{stage}{player}: {action_label}")

        # Pre-calculated analysis (concise)
        opponent_range = context.get('opponent_range')
        hand_equity = context.get('hand_equity')

        if opponent_range:
            range_size = len(opponent_range)
            lines.append(f"Opp Range: {range_size} hands")

        if hand_equity is not None:
            lines.append(f"Your Equity: {hand_equity*100:.1f}%")

        if context.get('board_texture'):
            board_texture_data = context.get('board_texture')
            if isinstance(board_texture_data, dict):
                board_type_str = board_texture_data.get('type', '')
            else:
                board_type_str = str(board_texture_data)
            lines.append(f"Board: {board_type_str[:6]}")  # Truncate to save tokens

        lines.append("")
        lines.append("Legal Actions:")
        legal_actions = context.get('legal_actions', [])
        legal_actions_labels = context.get('legal_actions_labels', {})
        for action in legal_actions:
            label = legal_actions_labels.get(action, f'Action {action}')
            lines.append(f"- {label}")

        # Add clarification if present
        if 'clarification' in context:
            lines.append(f"Note: {context['clarification']}")

        return "\n".join(lines)
    
    def _build_system_prompt(self, opponent_position='big_blind'):
        """
        Build system prompt that guides LLM to reason step-by-step before making decisions.

        Args:
            opponent_position (str): The opponent's position ('button' for SB, 'big_blind' for BB)

        Returns:
            str: System prompt text
        """
        position_description = "BB (out of position)" if opponent_position == 'big_blind' else "SB (button, in position)"

        return f"""You are an expert NLHE poker player making GTO decisions.

**Decision Process:**
1. Analyze opponent range based on position, action history, and stack depth
2. Assess your hand equity vs opponent range (if available)
3. Evaluate pot odds and expected value
4. Consider position and stack depth implications
5. Select optimal action from legal options

**Core Principles:**
- Range-based thinking: Consider opponent ranges based on action history
- Pot odds vs equity: Call when equity > pot odds, fold when equity < pot odds
- Position: You are {position_description} - {"be conservative OOP" if opponent_position == 'big_blind' else "be aggressive IP"}
- Stack depth: <20BB aggressive, 20-50BB balanced, >100BB more postflop play

**Rules:**
- ALWAYS provide detailed step-by-step reasoning first
- ONLY select from legal actions provided
- Use select_poker_action tool with reasoning and action
- Consider pre-calculated values (range, equity, pot odds)
- Choose conservative action if analysis is unclear"""
    
    def _get_tool_calling_schema(self, legal_actions_labels=None, legal_actions=None):
        """
        Define tool calling schema for select_poker_action tool.
        Uses actual labeled actions (e.g., "3-bet to 10 BB") instead of generic types.

        Args:
            legal_actions_labels (dict): Dictionary mapping Action enums to their labels
                e.g., {Action.FOLD: 'Fold', Action.CHECK_CALL: 'Call', Action.RAISE_HALF_POT: '3-bet to 10 BB'}
            legal_actions (list, optional): List of legal Action enums to filter labels

        Returns:
            list: Tool schema list for OpenAI API
        """
        # If no labels provided, fall back to generic action types
        if not legal_actions_labels:
            base_actions = ["fold", "check", "call", "raise_half_pot", "raise_pot", "all_in"]
            description = "The type of action to take. Select from the available legal actions shown in the context."
        else:
            # Build enum from actual labels
            # Only include labels for actions that are actually legal
            action_labels = []
            seen_labels = set()
            
            # Filter to only include labels for legal actions
            for action, label in legal_actions_labels.items():
                # Only include if action is legal (if legal_actions list provided)
                if legal_actions is None or action in legal_actions:
                    if label and label not in seen_labels:
                        action_labels.append(label)
                        seen_labels.add(label)
            
            # Ensure we have at least some actions
            if not action_labels:
                # Fallback to generic actions if no labels found
                base_actions = ["fold", "check", "call", "raise_half_pot", "raise_pot", "all_in"]
                description = "The type of action to take. Select from the available legal actions shown in the context."
            else:
                base_actions = action_labels
                description = "The type of action to take. Select the EXACT label from the legal actions shown in the context (e.g., '3-bet to 10 BB', 'Call', 'Fold')."

        return [
            {
                "type": "function",
                "function": {
                    "name": "select_poker_action",
                    "description": "Select optimal poker action from legal options using GTO principles. Use the EXACT label from the legal actions list.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "REQUIRED: Detailed step-by-step analysis following the decision process: 1) Opponent range assessment 2) Equity evaluation 3) Pot odds analysis 4) Position/stack considerations 5) Strategic conclusion"
                            },
                            "action_type": {
                                "type": "string",
                                "enum": base_actions,
                                "description": description + " - Select this action based on your detailed reasoning above."
                            }
                        },
                        "required": ["reasoning", "action_type"]
                    }
                }
            }
        ]
    
    def _call_llm_for_decision(self, context, retry_count=0):
        """
        Call LLM API with tool calling to get opponent decision.
        LLM now provides detailed reasoning first, then selects action.
        Includes retry logic with exponential backoff.

        Args:
            context (dict): Context dictionary
            retry_count (int): Current retry attempt (0 = first attempt)

        Returns:
            str: Action string from LLM (e.g., "fold", "call", "raise_half_pot"), or None on failure
        """
        max_retries = 2
        backoff_delays = [1, 2]  # Exponential backoff: 1s, 2s
        
        try:
            # Build system prompt
            opponent_position = context.get('opponent_position', 'big_blind')
            system_prompt = self._build_system_prompt(opponent_position)
            
            # Format context for prompt
            formatted_context = self._format_context_for_prompt(context)
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_context}
            ]
            
            # Get tool calling schema (pass legal_actions_labels for actual labeled actions)
            legal_actions_labels = context.get('legal_actions_labels', {})
            legal_actions = context.get('legal_actions', [])
            tools = self._get_tool_calling_schema(legal_actions_labels=legal_actions_labels, legal_actions=legal_actions)
            
            # Call LLM API with timeout
            try:
                def _make_api_call():
                    return self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": "select_poker_action"}},
                        max_tokens=2048  # Conservative token limit to avoid credit issues
                    )
                
                response = self.executor.submit(_make_api_call).result(timeout=self.executor_timeout)
                
                # Extract action from tool call
                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message
                    if message.tool_calls and len(message.tool_calls) > 0:
                        tool_call = message.tool_calls[0]
                        if tool_call.function.name == "select_poker_action":
                            import json
                            arguments = json.loads(tool_call.function.arguments)
                            action_type = arguments.get("action_type")
                            reasoning = arguments.get("reasoning", "")
                            
                            return action_type
                
                return None
                
            except FutureTimeoutError:
                if retry_count < max_retries:
                    delay = backoff_delays[retry_count] if retry_count < len(backoff_delays) else 2
                    time.sleep(delay)
                    return self._call_llm_for_decision(context, retry_count + 1)
                return None
                
        except Exception as e:
            if retry_count < max_retries:
                delay = backoff_delays[retry_count] if retry_count < len(backoff_delays) else 2
                time.sleep(delay)
                return self._call_llm_for_decision(context, retry_count + 1)
            return None
    
    def _map_llm_action_to_rlcard(self, llm_action, legal_actions, legal_actions_labels=None):
        """
        Map LLM tool call action (label string) to RLCard Action enum, with validation and fallback logic.
        
        Args:
            llm_action (str): Action label from LLM tool call (e.g., "3-bet to 10 BB", "Call", "Fold")
            legal_actions (list): List of legal RLCard Action enum values
            legal_actions_labels (dict, optional): Dictionary mapping Action enums to their labels
        
        Returns:
            int: RLCard Action enum value
        """
        # First, try to map using legal_actions_labels (preferred method)
        if legal_actions_labels:
            # Normalize strings for comparison (remove extra spaces, case-insensitive)
            def normalize_label(s):
                if not s:
                    return ""
                # Remove extra spaces, convert to lowercase
                return " ".join(s.lower().split())
            
            llm_action_normalized = normalize_label(llm_action)
            
            # Find the Action enum that matches this label
            for action, label in legal_actions_labels.items():
                if label:
                    label_normalized = normalize_label(label)
                    if label_normalized == llm_action_normalized:
                        if action in legal_actions:
                            return action
                        else:
                            pass
            
            # If exact match failed, try partial matching for common patterns
            # This handles cases where LLM might return slightly different formatting
            for action, label in legal_actions_labels.items():
                if label:
                    label_normalized = normalize_label(label)
                    # Check if the core action matches (e.g., "3-bet to 10 bb" matches "3-bet to 10BB")
                    # Extract key words from both labels
                    llm_words = set(llm_action_normalized.split())
                    label_words = set(label_normalized.split())
                    
                    # If most words match (at least 2 words in common for multi-word labels)
                    if len(llm_words) >= 2 and len(label_words) >= 2:
                        common_words = llm_words.intersection(label_words)
                        if len(common_words) >= 2:  # At least 2 words match
                            if action in legal_actions:
                                return action
        
        # Fallback: try generic action type mapping (for backward compatibility)
        action_map = {
            "fold": Action.FOLD,
            "call": Action.CHECK_CALL,
            "check": Action.CHECK_CALL,
            "raise_half_pot": Action.RAISE_HALF_POT,
            "raise_pot": Action.RAISE_POT,
            "all_in": Action.ALL_IN
        }
        
        rlcard_action = action_map.get(llm_action.lower())
        
        # Validate action is legal
        if rlcard_action and rlcard_action in legal_actions:
            return rlcard_action
        
        # If action not found or illegal, try fallback logic
        if rlcard_action not in legal_actions:
            # Fallback Priority (try alternatives before giving up):
            # 1. If raise_half_pot illegal but raise_pot legal → use raise_pot
            if llm_action.lower() in ["raise_half_pot", "raise to 3 bb", "3-bet to 10 bb", "4-bet to 25 bb", "bet ½ pot"]:
                if Action.RAISE_POT in legal_actions:
                    return Action.RAISE_POT
                elif Action.RAISE_HALF_POT in legal_actions:
                    return Action.RAISE_HALF_POT
            # 2. If raise_pot illegal but raise_half_pot legal → use raise_half_pot
            elif llm_action.lower() in ["raise_pot", "raise pot", "bet ⅔ pot"]:
                if Action.RAISE_HALF_POT in legal_actions:
                    return Action.RAISE_HALF_POT
                elif Action.RAISE_POT in legal_actions:
                    return Action.RAISE_POT
            # 3. If call/check illegal but CHECK_CALL legal → use CHECK_CALL (shouldn't happen, but safe)
            elif llm_action.lower() in ["call", "check"] and Action.CHECK_CALL in legal_actions:
                return Action.CHECK_CALL
        
        # Last resort: return first legal action
        return legal_actions[0]

