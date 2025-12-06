"""
Chatbot Coach

Provides conversational AI coaching using OpenAI GPT-4 or OpenRouter.
Integrates game context and maintains conversation history.

Supports both OpenAI and OpenRouter APIs:
- OpenAI: Direct API access using OPENAI_API_KEY
- OpenRouter: Unified API access using OPEN_ROUTER_KEY (takes precedence if both are set)
"""

import os
import time
import json
# logging removed
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception as e:
    OPENAI_AVAILABLE = False
    OpenAI = None
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import numpy as np

# Load environment variables
try:
    load_dotenv()
except (PermissionError, FileNotFoundError):
    # .env file not accessible, continue with system environment variables
    pass

# Configure logging


class ChatbotCoach:
    """
    Provides conversational AI coaching using OpenAI GPT-4 or OpenRouter.
    
    Handles LLM API calls, context injection, and conversation history management.
    Supports both OpenAI and OpenRouter APIs, with OpenRouter taking precedence if available.
    """
    
    def __init__(self):
        """
        Initialize ChatbotCoach with OpenAI or OpenRouter client and system prompt.

        Supports both OpenAI and OpenRouter APIs. Uses LLM_PROVIDER environment variable
        to determine which provider to use. Defaults to OpenAI if not specified.
        """
        # Check LLM_PROVIDER environment variable
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

        # Get API keys
        openrouter_key = os.getenv("OPEN_ROUTER_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        self.use_openrouter = False
        self.model = "gpt-4-turbo-preview"  # Default model
        self.llm_provider = llm_provider

        # Configure timeout for API calls (20 seconds for coach to handle complex analysis)
        self.api_timeout = 20.0

        # Initialize card mapping for converting string cards to indices
        self._init_card_mapping()

        # Validate provider choice and initialize appropriate client
        if llm_provider == "openrouter":
            if not openrouter_key or openrouter_key == "your_openrouter_key_here" or openrouter_key == "your_key_here":
                self.client = None
                self.api_key_available = False
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
                except Exception as e:
                    self.client = None
                    self.api_key_available = False

        elif llm_provider == "openai":
            if not OPENAI_AVAILABLE:
                self.client = None
                self.api_key_available = False
            elif not openai_key or openai_key == "your_openai_api_key_here" or openai_key == "your_key_here":
                self.client = None
                self.api_key_available = False
            else:
                try:
                    self.client = OpenAI(
                        api_key=openai_key,
                        timeout=self.api_timeout
                    )
                    self.api_key_available = True
                    self.model = "gpt-4-turbo-preview"
                except Exception as e:
                    self.client = None
                    self.api_key_available = False

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
                except Exception as e:
                    self.client = None
                    self.api_key_available = False
            else:
                self.client = None
                self.api_key_available = False
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
        
        # Conversation history storage: {session_id: [messages]}
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        
        # Thread pool executor for non-blocking API calls
        self.executor = ThreadPoolExecutor(max_workers=5)

    def _init_card_mapping(self):
        """
        Initialize card2index mapping for converting card strings to indices.
        This allows the coach to handle both string cards ('SK', 'C5') and integer indices.
        """
        # Try to load from RLCard first, fallback to manual mapping
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

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for poker coach persona.
        
        Returns:
            str: System prompt text
        """
        return """You are an expert poker coach specializing in heads-up No Limit Texas Hold'em. Your responses must follow this exact structure and be no more than 300 words total:

**1. SITUATION SUMMARY** (First paragraph)
- State the user's hole cards
- Describe the board (if any cards are visible)
- Identify the user's position (button/SB or big blind)
- Summarize previous action (who acted, what they did, bet sizes)
- Note relevant context: pot size, stack depths, SPR (stack-to-pot ratio)

**2. THOUGHT PROCESS** (Middle section)
Walk through the decision-making process step-by-step:
- Analyze opponent's likely range based on their actions
- Calculate pot odds and required equity (if facing a bet)
- Evaluate your hand's equity vs opponent's range
- Consider position, stack depth, and board texture
- Apply GTO principles: what would optimal play look like here?
- Weigh the factors: what makes this decision easy or difficult?

**3. RECOMMENDED ACTION** (Final paragraph)
- Clearly state the recommended action (Fold, Call, or Raise to X BB)
- Briefly explain why this is the best decision
- Note any important considerations or alternatives

**IMPORTANT FORMATTING REQUIREMENTS:**
- Include a blank line (paragraph break) between sections 1, 2, and 3
- Each section must be separated by an empty line for readability
- Do not run sections together without breaks

**Guidelines:**
- Maximum 300 words total
- Be concise but educational
- Use specific calculations when relevant (pot odds, equity percentages)
- Use consistent terminology ("continuation bet" not "c-bet", "pot odds" not "odds")
- Friendly but professional tone
- Focus on teaching the thinking process
- Use markdown formatting (bold, italic) for clarity

You have access to current game state and hand history when provided. Use this context to give accurate, situation-specific advice."""
    
    def chat(self, message: str, game_context: Optional[Dict[str, Any]] = None, 
             hand_history: Optional[List[Dict[str, Any]]] = None,
             session_id: str = "default") -> Dict[str, str]:
        """
        Chat with AI coach.
        
        Args:
            message: User's message/question
            game_context: Optional current game state for context injection
            hand_history: Optional list of previous hand decisions for context
            session_id: Session identifier for conversation history
        
        Returns:
            dict: {
                'response': str,  # Coach's response
                'timestamp': str   # ISO8601 timestamp
            }
        """
        if not self.api_key_available or not self.client:
            # Try rule-based fallback first
            rule_based_response = self._generate_rule_based_response(
                message, game_context, hand_history
            )
            if rule_based_response:
                return rule_based_response
            # If rule-based fails, use error fallback
            return self._generate_fallback_response(
                message, game_context,
                error_type="api_unavailable",
                error_details="API key not configured or client not initialized"
            )
        
        try:
            # Get conversation history
            history = self._get_conversation_history(session_id)

            # [AI_COACH_LOGGING] Log hand history retrieval from storage

            if hand_history:
                # Log sample of hand history structure for debugging
                sample_decisions = hand_history[-3:] if len(hand_history) > 3 else hand_history
                for i, decision in enumerate(sample_decisions):
                    player_id = decision.get('player_id', '?')
                    action = decision.get('action', '?')
                    stage = decision.get('stage', '?')
                    hand = decision.get('hand', [])

            # Build messages array
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]

            # Add conversation history
            messages.extend(history)
            
            # Inject game context and hand history if available and relevant
            context_text = self._inject_context(game_context, hand_history, message)
            if context_text:
                messages.append({
                    "role": "system",
                    "content": f"Current game context:\n{context_text}"
                })
            elif hand_history and len(hand_history) > 0:
                # This should not happen - hand history should always be included
                # Force include hand history as fallback
                try:
                    simple_context = self._format_hand_history_simple(hand_history)
                    if simple_context:
                        messages.append({
                            "role": "system",
                            "content": f"Most Recent Hand:\n{simple_context}"
                        })
                except Exception as e:
                    pass
            
            # Add user message
            messages.append({"role": "user", "content": message})
            
            # Determine if this is a hand analysis question (needs longer response)
            message_lower = message.lower()
            is_hand_analysis = any(keyword in message_lower for keyword in [
                "was that", "good call", "good fold", "good raise", "bad call", "bad fold", 
                "bad raise", "should i have", "was my", "decision", "hand", "play"
            ])
            
            # Call OpenAI API with timeout (15 seconds total, but API call itself has 12s timeout)
            try:
                response = self.executor.submit(
                    self._call_openai_api,
                    messages,
                    is_hand_analysis  # Pass flag for token adjustment
                ).result(timeout=21.0)
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Format response
                formatted_response = self._format_response(response_text, is_hand_analysis)
                
                # Add to conversation history
                self._add_to_history(session_id, "user", message)
                self._add_to_history(session_id, "assistant", response_text)
                
                return formatted_response
                
            except FutureTimeoutError:
                error_msg = f"OpenAI API call timed out after {self.api_timeout} seconds"
                # Try intelligent rule-based fallback first
                rule_based_response = self._generate_rule_based_response(
                    message, game_context, hand_history
                )
                if rule_based_response:
                    return rule_based_response
                # If rule-based fails, use error fallback
                return self._generate_fallback_response(
                    message, game_context, 
                    error_type="timeout",
                    error_details=error_msg,
                    timeout=True
                )
            except Exception as e:
                error_msg = f"OpenAI API call failed: {str(e)}"
                # Retry with exponential backoff (max 2 retries)
                try:
                    return self._retry_with_backoff(messages, message, game_context, session_id, hand_history)
                except Exception as retry_error:
                    # Try intelligent rule-based fallback first
                    rule_based_response = self._generate_rule_based_response(
                        message, game_context, hand_history
                    )
                    if rule_based_response:
                        return rule_based_response
                    # If rule-based fails, use error fallback
                    return self._generate_fallback_response(
                        message, game_context,
                        error_type="api_error",
                        error_details=f"API call failed after retries: {str(retry_error)}"
                    )
                
        except Exception as e:
            error_msg = f"Error in chat method: {str(e)}"
            # Try rule-based fallback first
            rule_based_response = self._generate_rule_based_response(
                message, game_context, hand_history
            )
            if rule_based_response:
                return rule_based_response
            # If rule-based fails, use error fallback
            return self._generate_fallback_response(
                message, game_context,
                error_type="general_error",
                error_details=error_msg
            )
    
    def _call_openai_api(self, messages: List[Dict[str, str]], is_hand_analysis: bool = False) -> Any:
        """
        Call OpenAI API with retry logic.
        
        Args:
            messages: List of message dicts for API call
            is_hand_analysis: Whether this is a hand analysis question (needs longer response)
        
        Returns:
            OpenAI API response
        """
        max_retries = 2
        retry_delays = [1.0, 2.0]  # Exponential backoff: 1s, 2s
        
        # Adjust max_tokens to ensure responses stay under 300 words (1 token ≈ 0.75 words, so 400 tokens ≈ 300 words)
        max_tokens = 400  # Maximum 300 words for all responses
        
        for attempt in range(max_retries + 1):
            try:
                # Log prompt size for debugging
                total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                return response
            except Exception as e:
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    time.sleep(delay)
                else:
                    raise
    
    def _retry_with_backoff(self, messages: List[Dict[str, str]], 
                           user_message: str, game_context: Optional[Dict[str, Any]],
                           session_id: str, hand_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
        """
        Retry API call with exponential backoff.
        
        Args:
            messages: Messages for API call
            user_message: Original user message
            game_context: Game context
            session_id: Session ID
            hand_history: Optional hand history for fallback
        
        Returns:
            dict: Response or fallback
        """
        max_retries = 2
        retry_delays = [1.0, 2.0]
        
        # Determine if this is a hand analysis question
        message_lower = user_message.lower()
        is_hand_analysis = any(keyword in message_lower for keyword in [
            "was that", "good call", "good fold", "good raise", "bad call", "bad fold", 
            "bad raise", "should i have", "was my", "decision", "hand", "play"
        ])
        
        for attempt in range(max_retries):
            try:
                time.sleep(retry_delays[attempt])
                response = self._call_openai_api(messages, is_hand_analysis)
                response_text = response.choices[0].message.content
                formatted_response = self._format_response(response_text, is_hand_analysis)
                
                # Add to conversation history
                self._add_to_history(session_id, "user", user_message)
                self._add_to_history(session_id, "assistant", response_text)
                
                return formatted_response
            except Exception as e:
                pass
        
        # All retries failed, try rule-based fallback first
        rule_based_response = self._generate_rule_based_response(
            user_message, game_context, hand_history
        )
        if rule_based_response:
            return rule_based_response
        
        # If rule-based fails, return error fallback
        return self._generate_fallback_response(
            user_message, game_context,
            error_type="retry_failed",
            error_details="All retry attempts failed"
        )
    
    def _inject_context(self, game_context: Optional[Dict[str, Any]], 
                       hand_history: Optional[List[Dict[str, Any]]],
                       message: str) -> Optional[str]:
        """
        Inject game context and hand history into prompt when relevant.
        
        Args:
            game_context: Current game state
            hand_history: List of previous hand decisions
            message: User's message to determine relevance
        
        Returns:
            str: Formatted context text, or None if not relevant
        """

        message_lower = message.lower()
        
        # Determine if current game context is relevant based on message
        context_relevant_keywords = [
            "current", "now", "this hand", "this situation", "what should",
            "optimal play", "should i"
        ]
        context_relevant = any(keyword in message_lower for keyword in context_relevant_keywords)


        context_parts = []
        
        # Add current game context if available and relevant
        if game_context and context_relevant:
            try:
                # Extract relevant game state
                hand = game_context.get('hand', [])
                board = game_context.get('public_cards', [])
                pot = game_context.get('pot', 0)
                big_blind = game_context.get('big_blind', 2)
                all_chips = game_context.get('all_chips', [0, 0])
                
                # Format hand
                hand_str = self._format_cards(hand) if hand else "Unknown"
                
                # Format board
                board_str = self._format_cards(board) if board else "No board cards yet"
                
                # Calculate stack depths
                player_chips = all_chips[0] if len(all_chips) > 0 else 0
                opponent_chips = all_chips[1] if len(all_chips) > 1 else 0
                player_stack_bb = player_chips / big_blind if big_blind > 0 else 0
                opponent_stack_bb = opponent_chips / big_blind if big_blind > 0 else 0
                effective_stack = min(player_chips, opponent_chips)
                effective_stack_bb = effective_stack / big_blind if big_blind > 0 else 0
                
                # Calculate SPR (stack-to-pot ratio)
                pot_bb = pot / big_blind if big_blind > 0 else pot
                spr = effective_stack / pot if pot > 0 and big_blind > 0 else 0
                
                # Determine position (simplified)
                button_id = game_context.get('button_id', 0)
                current_player = game_context.get('current_player', 0)
                position = "button" if current_player == button_id else "big_blind"
                
                context_parts.extend([
                    "Current Game State:",
                    f"  Hand: {hand_str}",
                    f"  Board: {board_str}",
                    f"  Pot: {pot_bb:.1f} BB" if big_blind > 0 else f"  Pot: {pot}",
                    f"  Position: {position}",
                    f"  Stack Depths: Player {player_stack_bb:.1f} BB, Opponent {opponent_stack_bb:.1f} BB",
                    f"  Effective Stack: {effective_stack_bb:.1f} BB",
                    f"  SPR (Stack-to-Pot Ratio): {spr:.1f}" if spr > 0 else ""
                ])
            except Exception as e:
                pass
        
        # Always include most recent hand history if available
        if hand_history and len(hand_history) > 0:
            hand_history_added = False

            # First, try to get player decisions to verify we have data
            # Handle both int and string player_id values
            player_decisions_count = 0
            for d in hand_history:
                player_id = d.get('player_id')
                if player_id == 0 or player_id == "0" or str(player_id) == "0":
                    player_decisions_count += 1

            
            if player_decisions_count == 0:
                # Still add a note that hand history exists
                context_parts.append("\nMost Recent Hand:")
                context_parts.append(f"  (Hand history available with {len(hand_history)} total decisions, but no player decisions found)")
                hand_history_added = True
            else:
                try:

                    # Get the most recent hand by finding the last preflop decision and all decisions after it
                    # This is more reliable than grouping by stage resets
                    most_recent_hand_decisions = []

                    # Extract the most recent complete hand by looking for stage transitions
                    # A hand typically progresses through stages: preflop(0) -> flop(1) -> turn(2) -> river(3)
                    # Work backwards to find where the current hand began

                    most_recent_hand_decisions = []
                    current_stage = None
                    hand_start_index = 0

                    # Work backwards through decisions to find hand boundaries
                    for i in range(len(hand_history) - 1, -1, -1):
                        decision = hand_history[i]
                        stage = decision.get('stage', 0)
                        # Handle both int and string stage values
                        if isinstance(stage, str):
                            stage = int(stage) if stage.isdigit() else 0
                        elif not isinstance(stage, int):
                            stage = 0

                        # If we encounter a stage that's earlier than current, we've found a hand boundary
                        if current_stage is None:
                            current_stage = stage
                        elif stage < current_stage:
                            # Stage went backwards, this indicates start of new hand
                            hand_start_index = i + 1
                            break

                    # Extract decisions from the start of this hand
                    most_recent_hand_decisions = hand_history[hand_start_index:]

                    # If we still don't have enough context, fall back to last 10 decisions
                    if len(most_recent_hand_decisions) < 3:
                        most_recent_hand_decisions = hand_history[-10:] if len(hand_history) > 10 else hand_history
                    
                    # Extract both player and opponent decisions for full hand context
                    # Player decisions (player_id == 0) for analysis focus, opponent decisions for context
                    player_decisions = []
                    opponent_decisions = []
                    for d in most_recent_hand_decisions:
                        player_id = d.get('player_id')
                        # Handle both int and string player_id
                        if player_id == 0 or player_id == "0" or str(player_id) == "0":
                            player_decisions.append(d)
                        else:
                            # Include opponent decisions for context
                            opponent_decisions.append(d)


                    if player_decisions:
                        context_parts.append("\nMost Recent Hand:")

                        # Combine and sort all decisions by their order in the hand
                        all_hand_decisions = []
                        for d in most_recent_hand_decisions:
                            all_hand_decisions.append(d)

                        # Sort by the order they appear in most_recent_hand_decisions (maintains chronological order)
                        # Show both player and opponent actions for full context
                        for j, decision in enumerate(all_hand_decisions, 1):
                            try:
                                player_id = decision.get('player_id')
                                action = decision.get('action', 'Unknown')
                                stage = decision.get('stage', 0)
                                hand_cards = decision.get('hand', [])
                                board_cards = decision.get('public_cards', [])


                                # Convert numpy types and map to readable format
                                action = self._convert_numpy_type(action)
                                stage = self._convert_numpy_type(stage)

                                action_map = {0: "Fold", 1: "Check/Call", 2: "Raise ½ Pot", 3: "Raise Pot", 4: "All-In"}
                                stage_map = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}

                                action_name = action_map.get(int(action) if isinstance(action, (int, float)) else action, f"Action {action}")
                                stage_name = stage_map.get(int(stage) if isinstance(stage, (int, float)) else stage, f"Stage {stage}")

                                # Format cards
                                hand_str = self._format_cards(hand_cards) if hand_cards else ""
                                board_str = self._format_cards(board_cards) if board_cards else ""

                                # Get position if available
                                position = decision.get('position')
                                position_str = f", Position: {position}" if position else ""

                                # Determine if this is a player or opponent action
                                is_player_action = (player_id == 0 or player_id == "0" or str(player_id) == "0")
                                player_label = "You" if is_player_action else "Opp"


                                # Build decision text with player/opponent distinction
                                if j == 1 and hand_str and is_player_action:
                                    # First player decision: include hole cards and position
                                    decision_text = f"  {j}. {stage_name}: {player_label} {action_name} (Hand: {hand_str}{position_str})"
                                elif j == len(all_hand_decisions) and board_str:
                                    # Last decision: include final board
                                    decision_text = f"  {j}. {stage_name}: {player_label} {action_name} (Board: {board_str})"
                                else:
                                    # Regular decisions: include hand cards for opponent actions to show what they had
                                    hand_info = f" (Hand: {hand_str})" if hand_str and not is_player_action else ""
                                    decision_text = f"  {j}. {stage_name}: {player_label} {action_name}{hand_info}"

                                context_parts.append(decision_text)
                            except Exception as e:
                                player_label = "You" if (decision.get('player_id') == 0 or decision.get('player_id') == "0" or str(decision.get('player_id')) == "0") else "Opp"
                                context_parts.append(f"  {j}. {player_label}: {decision.get('action', 'Unknown')}")
                        hand_history_added = True
                    else:
                        # Fallback: show recent decisions including both player and opponent actions
                        context_parts.append("\nRecent Hand History:")
                        recent_decisions = hand_history[-10:]  # Get last 10 decisions for context
                        if recent_decisions:
                            for i, decision in enumerate(recent_decisions, 1):
                                try:
                                    player_id = decision.get('player_id')
                                    action = decision.get('action', 'Unknown')
                                    stage = decision.get('stage', 0)

                                    # Convert numpy types to Python native types
                                    action = self._convert_numpy_type(action)
                                    stage = self._convert_numpy_type(stage)

                                    action_map = {0: "Fold", 1: "Check/Call", 2: "Raise ½ Pot", 3: "Raise Pot", 4: "All-In"}
                                    stage_map = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}

                                    # Convert to int if possible
                                    try:
                                        action_int = int(action) if isinstance(action, (int, float)) else action
                                    except (ValueError, TypeError):
                                        action_int = action

                                    try:
                                        stage_int = int(stage) if isinstance(stage, (int, float)) else stage
                                    except (ValueError, TypeError):
                                        stage_int = stage

                                    hand_cards = decision.get('hand', [])
                                    board_cards = decision.get('public_cards', [])
                                    position = decision.get('position')
                                    hand_str = self._format_cards(hand_cards) if hand_cards else "Unknown"
                                    board_str = self._format_cards(board_cards) if board_cards else "No board"
                                    position_str = f", Position: {position}" if position else ""

                                    # Determine if this is a player or opponent action
                                    is_player_action = (player_id == 0 or player_id == "0" or str(player_id) == "0")
                                    player_label = "You" if is_player_action else "Opp"

                                    context_parts.append(
                                        f"  {i}. {stage_map.get(stage_int, 'Unknown')}: {player_label} {action_map.get(action_int, 'Unknown')} "
                                        f"(Hand: {hand_str}, Board: {board_str}{position_str})"
                                    )
                                except Exception as e:
                                    context_parts.append(f"  {i}. Decision: {decision}")
                            hand_history_added = True
                        else:
                            context_parts.append("  (No recent player decisions found)")
                            hand_history_added = True
                except Exception as e:
                    # Fallback: show simple list of recent decisions
                    try:
                        context_parts.append("\nMost Recent Hand (simplified):")
                        recent_player_decisions = []
                        for d in hand_history[-10:]:
                            player_id = d.get('player_id')
                            if player_id == 0 or player_id == "0" or str(player_id) == "0":
                                recent_player_decisions.append(d)
                        if recent_player_decisions:
                            for i, decision in enumerate(recent_player_decisions, 1):
                                try:
                                    action = decision.get('action', 'Unknown')
                                    stage = decision.get('stage', 0)
                                    
                                    # Convert numpy types to Python native types
                                    action = self._convert_numpy_type(action)
                                    stage = self._convert_numpy_type(stage)
                                    
                                    action_map = {0: "Fold", 1: "Check/Call", 2: "Raise ½ Pot", 3: "Raise Pot", 4: "All-In"}
                                    stage_map = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}
                                    
                                    # Convert to int if possible
                                    try:
                                        action_int = int(action) if isinstance(action, (int, float)) else action
                                    except (ValueError, TypeError):
                                        action_int = action
                                    
                                    try:
                                        stage_int = int(stage) if isinstance(stage, (int, float)) else stage
                                    except (ValueError, TypeError):
                                        stage_int = stage
                                    
                                    hand_cards = decision.get('hand', [])
                                    board_cards = decision.get('public_cards', [])
                                    position = decision.get('position')
                                    hand_str = self._format_cards(hand_cards) if hand_cards else "Unknown"
                                    board_str = self._format_cards(board_cards) if board_cards else "No board"
                                    position_str = f", Position: {position}" if position else ""
                                    context_parts.append(
                                        f"  {i}. {stage_map.get(stage_int, 'Unknown')}: {action_map.get(action_int, 'Unknown')} "
                                        f"(Hand: {hand_str}, Board: {board_str}{position_str})"
                                    )
                                except Exception as e:
                                    context_parts.append(f"  {i}. Decision: {decision}")
                            hand_history_added = True
                    except Exception as e2:
                        pass
            
            # Ensure hand history was added - if not, add a minimal version
            if not hand_history_added:
                context_parts.append("\nMost Recent Hand:")
                context_parts.append(f"  (Hand history available with {len(hand_history)} decisions, but formatting failed)")
        
        # Always return context if we have hand history or other context
        if not context_parts:
            # If we have hand history but couldn't format it, still include a basic note
            if hand_history and len(hand_history) > 0:
                # Include a basic hand history note
                context_parts.append("Most Recent Hand:")
                context_parts.append(f"  (Hand history available with {len(hand_history)} decisions)")
                # Try to include at least the last few decisions in raw format
                try:
                    last_decisions = hand_history[-5:]
                    for i, decision in enumerate(last_decisions, 1):
                        player_id = decision.get('player_id', '?')
                        action = decision.get('action', '?')
                        stage = decision.get('stage', '?')
                        context_parts.append(f"  {i}. Player {player_id}, Stage {stage}, Action {action}")
                except Exception as e:
                    pass
            else:
                return None
        
        result = "\n".join(context_parts)

        # Log the context being sent for debugging

        # Limit total context size to prevent overly long prompts (max 2000 chars)
        # This helps prevent timeout issues with very large prompts
        max_context_length = 2000
        if len(result) > max_context_length:
            result = result[:max_context_length] + "\n... (context truncated for performance)"

        return result
    
    def _format_hand_history_simple(self, hand_history: List[Dict[str, Any]]) -> Optional[str]:
        """
        Simple fallback method to format hand history.
        
        Args:
            hand_history: List of hand decisions
        
        Returns:
            str: Formatted hand history or None
        """
        if not hand_history or len(hand_history) == 0:
            return None
        
        try:
            # Get last 10 player decisions
            recent_player_decisions = [
                d for d in hand_history[-10:] if d.get('player_id') == 0
            ]
            
            if not recent_player_decisions:
                return None
            
            parts = []
            for i, decision in enumerate(recent_player_decisions, 1):
                action = decision.get('action', 'Unknown')
                stage = decision.get('stage', 0)
                hand_cards = decision.get('hand', [])
                board_cards = decision.get('public_cards', [])
                pot = decision.get('pot', 0)
                position = decision.get('position')

                action_map = {0: "Fold", 1: "Check/Call", 2: "Raise ½ Pot", 3: "Raise Pot", 4: "All-In"}
                stage_map = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}

                action_name = action_map.get(action, f"Action {action}")
                stage_name = stage_map.get(stage, f"Stage {stage}")
                hand_str = self._format_cards(hand_cards) if hand_cards else "Unknown"
                board_str = self._format_cards(board_cards) if board_cards else "No board"
                position_str = f", Position: {position}" if position else ""

                # Debug logging for hand cards
                if hand_cards:
                    print(f"[CHATBOT_DEBUG] Decision {i}: player_id={decision.get('player_id')}, hand_cards={hand_cards}, formatted={hand_str}")

                parts.append(
                    f"  {i}. {stage_name}: {action_name} "
                    f"(Your hand: {hand_str}, Board: {board_str}, Pot: {pot}{position_str})"
                )
            
            return "\n".join(parts)
        except Exception as e:
            return None
    
    def _convert_numpy_type(self, value: Any) -> Any:
        """
        Safely convert numpy types to Python native types.
        
        Args:
            value: Value that might be a numpy type
        
        Returns:
            Python native type (int, float, or original value)
        """
        try:
            # Import numpy at function level to avoid scoping issues
            import numpy as np_local
            if isinstance(value, (np_local.integer, np_local.int64, np_local.int32)):
                return int(value)
            elif isinstance(value, (np_local.floating, np_local.float64)):
                return float(value)
            else:
                return value
        except (NameError, AttributeError, ImportError, TypeError):
            # numpy not available, types don't exist, or conversion failed - return as-is
            # Try basic int/float conversion as fallback
            try:
                if isinstance(value, (int, float)):
                    return int(value) if isinstance(value, (int, float)) and not isinstance(value, float) else float(value)
                return value
            except (ValueError, TypeError):
                return value
    
    def _format_cards(self, cards: List) -> str:
        """
        Format card indices or strings to readable string.

        Args:
            cards: List of card indices from RLCard (0-51) or card strings ('SK', 'C5')

        Returns:
            str: Formatted card string (e.g., "A♠ K♥")
        """
        if not cards:
            return ""

        rank_names = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        suit_names = ['s', 'h', 'd', 'c']  # spades, hearts, diamonds, clubs
        suit_symbols = ['♠', '♥', '♦', '♣']

        formatted_cards = []
        for card in cards:
            if isinstance(card, str):
                # Handle string cards like 'SK', 'C5' (Suit first, then Rank)
                if len(card) == 2:
                    suit_char, rank_char = card[0], card[1]
                    # Map rank character to index
                    rank_map = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 0}
                    suit_map = {'S': 0, 'H': 1, 'D': 2, 'C': 3}

                    rank = rank_map.get(rank_char.upper(), 0)
                    suit = suit_map.get(suit_char.upper(), 0)
                    formatted_cards.append(f"{rank_names[rank]}{suit_symbols[suit]}")
                else:
                    # Try to convert using card2index mapping
                    if hasattr(self, 'card2index') and card in self.card2index:
                        card_index = self.card2index[card]
                        rank = card_index % 13
                        suit = card_index // 13
                        formatted_cards.append(f"{rank_names[rank]}{suit_symbols[suit]}")
                    else:
                        # Fallback: use card string as-is
                        formatted_cards.append(card)
            else:
                # Handle integer card indices (original logic)
                try:
                    card_int = int(card)
                    rank = card_int % 13
                    suit = card_int // 13
                    formatted_cards.append(f"{rank_names[rank]}{suit_symbols[suit]}")
                except (ValueError, TypeError):
                    # Safe fallback
                    formatted_cards.append(str(card))

        return " ".join(formatted_cards)
    
    def _get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            list: List of message dicts (last 10 messages)
        """
        return self.conversation_history.get(session_id, [])
    
    def _add_to_history(self, session_id: str, role: str, content: str):
        """
        Add message to conversation history.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        # Add message
        self.conversation_history[session_id].append({
            "role": role,
            "content": content
        })
        
        # Keep only last 10 messages
        if len(self.conversation_history[session_id]) > 10:
            self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
    
    def _format_response(self, response_text: str, is_hand_analysis: bool = False, is_error: bool = False) -> Dict[str, str]:
        """
        Format LLM response with timestamp.
        
        Args:
            response_text: Raw response from LLM
            is_hand_analysis: Whether this is a hand analysis question (allows longer response)
            is_error: Whether this is an error response (should not be truncated)
        
        Returns:
            dict: {
                'response': str,
                'timestamp': str (ISO8601)
            }
        """
        # Don't truncate error messages - they contain important troubleshooting information
        if not is_error:
            # Limit word count to 300 words maximum for all responses
            words = response_text.strip().split()
            max_words = 300
            if len(words) > max_words:
                # Try to truncate at a paragraph break to preserve structure
                truncated_text = ' '.join(words[:max_words])
                # Look for the last complete paragraph (ending with double newline or section marker)
                lines = response_text.split('\n')
                word_count = 0
                truncated_lines = []

                for line in lines:
                    line_words = line.strip().split()
                    if word_count + len(line_words) <= max_words:
                        truncated_lines.append(line)
                        word_count += len(line_words)
                    else:
                        # If we can't fit the whole line, truncate within the line
                        remaining_words = max_words - word_count
                        if remaining_words > 0:
                            truncated_line = ' '.join(line_words[:remaining_words])
                            truncated_lines.append(truncated_line)
                        break

                # Join lines back together, preserving structure
                response_text = '\n'.join(truncated_lines)
        
        return {
            'response': response_text.strip(),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    def _generate_rule_based_response(self, message: str,
                                     game_context: Optional[Dict[str, Any]] = None,
                                     hand_history: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, str]]:
        """
        Generate intelligent rule-based response when API is unavailable.
        
        Provides useful responses for common poker questions using pattern matching
        and available game context.
        
        Args:
            message: User's message
            game_context: Optional current game state
            hand_history: Optional hand history
            
        Returns:
            dict: Rule-based response, or None if unable to generate useful response
        """
        try:
            if not message or not isinstance(message, str):
                return None
            
            message_lower = message.lower().strip()
            
            # Pattern matching for common question types
            response_parts = []
            
            # Hand analysis questions
            if any(keyword in message_lower for keyword in [
                "was that", "good call", "good fold", "good raise", "bad call", "bad fold",
                "bad raise", "should i have", "was my", "decision", "hand", "play"
            ]):
                response_parts.append("**Hand Analysis**")
                
                # Try to analyze the most recent decision from hand history
                if hand_history and len(hand_history) > 0:
                    # Get last player decision
                    player_decisions = [
                        d for d in hand_history[-10:] 
                        if d.get('player_id') == 0 or d.get('player_id') == "0" or str(d.get('player_id')) == "0"
                    ]
                    
                    if player_decisions:
                        last_decision = player_decisions[-1]
                        action = last_decision.get('action', 'Unknown')
                        stage = last_decision.get('stage', 0)
                        hand_cards = last_decision.get('hand', [])
                        board_cards = last_decision.get('public_cards', [])
                        pot = last_decision.get('pot', 0)
                        
                        # Map action to readable format
                        action_map = {0: "Fold", 1: "Check/Call", 2: "Raise ½ Pot", 3: "Raise Pot", 4: "All-In"}
                        stage_map = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}
                        
                        action_int = int(action) if isinstance(action, (int, float)) else action
                        stage_int = int(stage) if isinstance(stage, (int, float)) else stage
                        
                        action_name = action_map.get(action_int, f"Action {action}")
                        stage_name = stage_map.get(stage_int, f"Stage {stage}")
                        
                        hand_str = self._format_cards(hand_cards) if hand_cards else "Unknown"
                        board_str = self._format_cards(board_cards) if board_cards else "No board"
                        
                        response_parts.append(f"Looking at your most recent decision: **{stage_name}: {action_name}**")
                        response_parts.append(f"Your hand: {hand_str}")
                        if board_cards:
                            response_parts.append(f"Board: {board_str}")
                        
                        # Provide general guidance based on action
                        if action_int == 0:  # Fold
                            response_parts.append("\n**General guidance on folding:**")
                            response_parts.append("- Folding is often correct when you have weak hands and face aggression")
                            response_parts.append("- Consider pot odds: if you need more equity than your hand has, folding is correct")
                            response_parts.append("- Position matters: easier to fold out of position when facing bets")
                        elif action_int == 1:  # Check/Call
                            response_parts.append("\n**General guidance on calling:**")
                            response_parts.append("- Calling is correct when you have enough equity vs opponent's range")
                            response_parts.append("- Calculate pot odds: pot size vs bet size tells you required equity")
                            response_parts.append("- Consider implied odds: if you hit your hand, can you win more?")
                        elif action_int in [2, 3, 4]:  # Raise
                            response_parts.append("\n**General guidance on raising:**")
                            response_parts.append("- Raising builds the pot with strong hands")
                            response_parts.append("- Can be used as a bluff with good board texture")
                            response_parts.append("- Consider stack depth: deeper stacks allow for more post-flop play")
                    else:
                        response_parts.append("I can see your hand history, but I need more context to analyze your specific decision.")
                else:
                    response_parts.append("To analyze your decision, I'd need to see your hand history. Try asking about a specific hand you just played.")
                
                # Add note for hand analysis questions
                response_parts.append("\n*Note: For detailed analysis, the AI coach is currently unavailable. This is a rule-based response.*")
            
            # Strategy questions
            elif any(keyword in message_lower for keyword in [
                "strategy", "gto", "optimal", "how to", "what should", "when to", "bet sizing",
                "position", "pot odds", "equity", "spr", "stack depth"
            ]):
                response_parts.append("**Strategy Guidance**")
                
                if "pot odds" in message_lower or "odds" in message_lower:
                    response_parts.append("**Pot Odds:**")
                    response_parts.append("- Pot odds = (Pot size) / (Bet amount + Pot size)")
                    response_parts.append("- Example: Pot is $10, opponent bets $5. Pot odds = 10/(5+10) = 66.7%")
                    response_parts.append("- You need at least 33.3% equity to call profitably")
                    response_parts.append("- Compare your hand's equity vs opponent's range to decide")
                
                if "position" in message_lower:
                    response_parts.append("\n**Position:**")
                    response_parts.append("- Acting last (button) gives you more information")
                    response_parts.append("- You can play more hands in position")
                    response_parts.append("- Out of position, play tighter and be more cautious")
                
                if "spr" in message_lower or "stack" in message_lower:
                    response_parts.append("\n**Stack-to-Pot Ratio (SPR):**")
                    response_parts.append("- SPR = Effective stack / Pot size")
                    response_parts.append("- Low SPR (< 3): More committed, less room for post-flop play")
                    response_parts.append("- High SPR (> 10): More room for maneuvering, can play for stacks")
                    response_parts.append("- Adjust your strategy based on SPR")
                
                if "bet sizing" in message_lower or "bet" in message_lower:
                    response_parts.append("\n**Bet Sizing:**")
                    response_parts.append("- Value bets: 50-75% pot with strong hands")
                    response_parts.append("- Bluffs: Similar sizing to value bets for balance")
                    response_parts.append("- Consider board texture: wet boards may need larger bets")
                
                response_parts.append("\n*Note: For detailed strategy advice, the AI coach is currently unavailable. This is a rule-based response.*")
            
            # General questions
            elif any(keyword in message_lower for keyword in [
                "hello", "hi", "help", "what can", "explain", "tell me"
            ]):
                response_parts.append("**Poker Coach Help**")
                response_parts.append("I can help you with:")
                response_parts.append("- Analyzing your hand decisions")
                response_parts.append("- Explaining GTO strategy concepts")
                response_parts.append("- Pot odds and equity calculations")
                response_parts.append("- Position and stack depth considerations")
                response_parts.append("- Bet sizing principles")
                response_parts.append("\nTry asking:")
                response_parts.append("- 'Was that a good call?' (analyze your last decision)")
                response_parts.append("- 'What are pot odds?' (explain a concept)")
                response_parts.append("- 'How should I play this hand?' (get strategy advice)")
                response_parts.append("\n*Note: The AI coach is currently unavailable. This is a rule-based response.*")
            
            # Current game state questions
            elif game_context and any(keyword in message_lower for keyword in [
                "current", "now", "this hand", "this situation", "what should i"
            ]):
                response_parts.append("**Current Situation Analysis**")
                
                hand = game_context.get('hand', [])
                board = game_context.get('public_cards', [])
                pot = game_context.get('pot', 0)
                big_blind = game_context.get('big_blind', 2)
                all_chips = game_context.get('all_chips', [0, 0])
                
                if hand:
                    hand_str = self._format_cards(hand)
                    response_parts.append(f"Your hand: {hand_str}")
                
                if board:
                    board_str = self._format_cards(board)
                    response_parts.append(f"Board: {board_str}")
                
                if pot and big_blind:
                    pot_bb = pot / big_blind
                    response_parts.append(f"Pot: {pot_bb:.1f} BB")
                
                if all_chips and len(all_chips) >= 2 and big_blind:
                    player_chips = all_chips[0]
                    opponent_chips = all_chips[1]
                    effective_stack = min(player_chips, opponent_chips)
                    spr = effective_stack / pot if pot > 0 else 0
                    response_parts.append(f"Effective stack: {effective_stack / big_blind:.1f} BB")
                    if spr > 0:
                        response_parts.append(f"SPR: {spr:.1f}")
                
                response_parts.append("\n**General considerations:**")
                response_parts.append("- Evaluate your hand strength vs the board")
                response_parts.append("- Consider your position and opponent's tendencies")
                response_parts.append("- Calculate pot odds if facing a bet")
                response_parts.append("- Think about your range and opponent's range")
                
                response_parts.append("\n*Note: For detailed analysis, the AI coach is currently unavailable. This is a rule-based response.*")
            
            # If we couldn't match any pattern, return None to use error fallback
            if not response_parts:
                return None
            
            response_text = "\n".join(response_parts)
            return self._format_response(response_text, is_hand_analysis=True, is_error=False)
        except Exception as e:
            return None
    
    def _generate_fallback_response(self, message: str, 
                                   game_context: Optional[Dict[str, Any]] = None,
                                   timeout: bool = False,
                                   error_type: Optional[str] = None,
                                   error_details: Optional[str] = None) -> Dict[str, str]:
        """
        Generate fallback response when LLM is unavailable.
        
        Args:
            message: User's message
            game_context: Optional game context
            timeout: Whether this is a timeout fallback
            error_type: Type of error that occurred
            error_details: Detailed error message
        
        Returns:
            dict: Fallback response with detailed error information
        """
        # Build detailed error explanation
        error_explanation_parts = []
        
        if error_type == "timeout" or timeout:
            error_explanation_parts.append("**Connection Timeout**")
            error_explanation_parts.append("The AI service took longer than 15 seconds to respond. This can happen when:")
            error_explanation_parts.append("- The AI service is experiencing high load")
            error_explanation_parts.append("- Your network connection is slow or unstable")
            error_explanation_parts.append("- The request is particularly complex")
            if error_details:
                error_explanation_parts.append(f"\nTechnical details: {error_details}")
        elif error_type == "api_error":
            error_explanation_parts.append("**API Error**")
            error_explanation_parts.append("The AI service returned an error. This could be due to:")
            error_explanation_parts.append("- Invalid API key or authentication issues")
            error_explanation_parts.append("- Service outage or maintenance")
            error_explanation_parts.append("- Rate limiting (too many requests)")
            if error_details:
                error_explanation_parts.append(f"\nTechnical details: {error_details}")
        elif error_type == "api_unavailable":
            error_explanation_parts.append("**Service Unavailable**")
            error_explanation_parts.append("The AI coach service is not properly configured:")
            error_explanation_parts.append("- API key may be missing or invalid")
            error_explanation_parts.append("- Please check your OPEN_ROUTER_KEY or OPENAI_API_KEY environment variable")
            if error_details:
                error_explanation_parts.append(f"\nTechnical details: {error_details}")
        elif error_type == "retry_failed":
            error_explanation_parts.append("**Connection Failed After Retries**")
            error_explanation_parts.append("Multiple attempts to connect to the AI service failed. This suggests:")
            error_explanation_parts.append("- Persistent network connectivity issues")
            error_explanation_parts.append("- The AI service may be temporarily unavailable")
            error_explanation_parts.append("- API rate limits may have been exceeded")
            if error_details:
                error_explanation_parts.append(f"\nTechnical details: {error_details}")
        elif error_type == "general_error":
            error_explanation_parts.append("**Unexpected Error**")
            error_explanation_parts.append("An unexpected error occurred while processing your request.")
            if error_details:
                error_explanation_parts.append(f"\nTechnical details: {error_details}")
        else:
            # Fallback for unknown errors
            error_explanation_parts.append("**Service Error**")
            error_explanation_parts.append("An error occurred while processing your request.")
            if error_details:
                error_explanation_parts.append(f"\nTechnical details: {error_details}")
        
        # Add troubleshooting steps
        error_explanation_parts.append("\n**What you can do:**")
        error_explanation_parts.append("1. Wait a moment and try your question again")
        error_explanation_parts.append("2. Check your internet connection")
        error_explanation_parts.append("3. If the problem persists, the service may be temporarily unavailable")
        
        # Add general poker strategy tip
        error_explanation_parts.append("\n**General Strategy Reminder:**")
        error_explanation_parts.append("While we work on the connection, remember that GTO (Game Theory Optimal) strategy focuses on:")
        error_explanation_parts.append("- **Balanced play**: Mixing your actions to avoid being predictable")
        error_explanation_parts.append("- **Pot odds**: Calculating whether a call is profitable based on the pot size and bet amount")
        error_explanation_parts.append("- **Position**: Acting last gives you more information and control")
        error_explanation_parts.append("- **Stack depth**: Deeper stacks allow for more post-flop play and bluffing opportunities")
        error_explanation_parts.append("- **SPR (Stack-to-Pot Ratio)**: Helps determine commitment level and optimal strategy")
        
        response_text = "\n".join(error_explanation_parts)
        
        return self._format_response(response_text, is_error=True)
    
    def clear_history(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
    
    def generate_hand_insights(self, hand_analysis: Dict[str, Any], 
                              game_state: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate LLM-powered key insights for hand analysis.
        
        Args:
            hand_analysis: Hand analysis from StrategyEvaluator
            game_state: Optional game state for context
        
        Returns:
            list: List of 3-5 key insights
        """
        if not self.api_key_available or not self.client:
            return []  # Fallback to rule-based insights only
        
        try:
            # Build prompt for insights generation
            decisions = hand_analysis.get('decisions', [])
            overall_grade = hand_analysis.get('overall_grade', 'C')
            overall_percentage = hand_analysis.get('overall_grade_percentage', 50)
            
            # Format decisions summary
            decisions_summary = []
            for i, decision in enumerate(decisions[:10], 1):  # Limit to 10 decisions
                stage = decision.get('stage', 'unknown')
                action = decision.get('action', 'Unknown')
                grade = decision.get('grade', 'C')
                decisions_summary.append(f"{i}. {stage}: {action} (Grade: {grade})")
            
            # Build prompt
            prompt = f"""Analyze this poker hand and generate 3-5 key insights:

Hand Context:
- Decisions: {chr(10).join(decisions_summary) if decisions_summary else 'No decisions'}
- Overall Grade: {overall_grade} ({overall_percentage}%)
- Key Moments: {len(decisions)} decision points analyzed

Generate insights that:
- Highlight good plays and mistakes
- Provide learning opportunities
- Are specific to this hand
- Are educational and actionable
- Use consistent terminology (e.g., "continuation bet" not "c-bet")
- Maintain friendly but professional tone
- Focus on learning and improvement

Format as a numbered list of concise insights (3-5 insights)."""
            
            # Call LLM with timeout
            try:
                response = self.executor.submit(
                    self._call_openai_api,
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                ).result(timeout=3.0)  # 3 second timeout
                
                response_text = response.choices[0].message.content
                
                # Parse insights (split by newlines or numbers)
                insights = self._parse_insights_list(response_text)
                
                return insights[:5]  # Limit to 5 insights
                
            except FutureTimeoutError:
                return []
            except Exception as e:
                return []
                
        except Exception as e:
            return []
    
    def enhance_explanation(self, rule_based_explanation: str, 
                           decision_context: Dict[str, Any],
                           game_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance Phase 1's rule-based explanation with LLM.
        
        Args:
            rule_based_explanation: Original rule-based explanation from Phase 1
            decision_context: Decision context (stage, action, grade, optimal_action)
            game_state: Optional game state for context
        
        Returns:
            str: Enhanced explanation (or original if LLM fails)
        """
        if not self.api_key_available or not self.client:
            return rule_based_explanation  # Fallback to original
        
        try:
            # Build prompt for explanation enhancement
            stage = decision_context.get('stage', 'unknown')
            action = decision_context.get('action', 'Unknown')
            grade = decision_context.get('grade', 'C')
            optimal_action = decision_context.get('optimal_action', 'Unknown')
            
            prompt = f"""You are an expert poker coach. Enhance this poker decision explanation with strategic reasoning and natural language flow.

Original Explanation:
{rule_based_explanation}

Decision Context:
- Stage: {stage}
- Action: {action}
- Grade: {grade}
- Optimal Action: {optimal_action}

Enhance the explanation by:
- Adding strategic reasoning and context (explain WHY the action is optimal/suboptimal)
- Improving natural language flow (make it read naturally, like a coach explaining to a student)
- Making it more educational (help the player understand the concept, not just the decision)
- Using consistent terminology (e.g., "continuation bet" not "c-bet", "pot odds" not "odds")
- Keeping a friendly but professional tone (supportive coach, not judgmental)
- DO NOT change the grade interpretation or accuracy
- Keep the same factual information
- Use 2-4 sentences for clarity

Return only the enhanced explanation, no additional commentary."""
            
            # Call LLM with timeout
            try:
                response = self.executor.submit(
                    self._call_openai_api,
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                ).result(timeout=3.0)  # 3 second timeout
                
                enhanced_explanation = response.choices[0].message.content.strip()
                
                # Validate that explanation is reasonable length
                if len(enhanced_explanation) > 2000:  # Too long, truncate
                    enhanced_explanation = enhanced_explanation[:2000] + "..."
                
                return enhanced_explanation
                
            except FutureTimeoutError:
                return rule_based_explanation
            except Exception as e:
                return rule_based_explanation
                
        except Exception as e:
            return rule_based_explanation
    
    def analyze_patterns(self, hand_history_list: List[Dict[str, Any]],
                        current_hand_analysis: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Analyze patterns across multiple hands using LLM.
        
        Args:
            hand_history_list: List of last 5-10 hands' decision summaries
            current_hand_analysis: Optional current hand analysis
        
        Returns:
            list: List of 2-4 pattern-based insights
        """
        if not self.api_key_available or not self.client:
            return []  # Fallback to empty list
        
        if not hand_history_list or len(hand_history_list) < 2:
            return []  # Need at least 2 hands for patterns
        
        try:
            # Build hand summaries for prompt
            hand_summaries = []
            for i, hand in enumerate(hand_history_list[-10:], 1):  # Last 10 hands
                decisions = hand.get('decisions', [])
                decision_count = len(decisions)
                if decision_count > 0:
                    # Summarize decisions
                    actions = [d.get('action', 'Unknown') for d in decisions[:5]]  # First 5 actions
                    summary = f"Hand {i}: {decision_count} decisions - {', '.join(actions)}"
                    hand_summaries.append(summary)
            
            # Add current hand if available
            current_summary = ""
            if current_hand_analysis:
                current_decisions = current_hand_analysis.get('decisions', [])
                if current_decisions:
                    actions = [d.get('action', 'Unknown') for d in current_decisions[:5]]
                    current_summary = f"\nCurrent Hand: {len(current_decisions)} decisions - {', '.join(actions)}"
            
            # Build prompt
            prompt = f"""Analyze these poker hands and identify patterns across multiple hands:

Last 5-10 Hands:
{chr(10).join(hand_summaries) if hand_summaries else 'No previous hands'}{current_summary}

Identify patterns such as:
- Consistent mistakes (e.g., "You consistently called with weak hands")
- Good habits (e.g., "You play position well")
- Tendencies (e.g., "You tend to overvalue second pair")
- Improvement areas

Generate 2-4 pattern-based insights that are:
- Accurate and helpful
- Actionable
- Specific to the patterns observed
- Use consistent terminology (e.g., "continuation bet" not "c-bet")
- Maintain friendly but professional tone
- Focus on learning and improvement

Format as a numbered list of concise insights."""
            
            # Call LLM with timeout
            try:
                response = self.executor.submit(
                    self._call_openai_api,
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                ).result(timeout=3.0)  # 3 second timeout
                
                response_text = response.choices[0].message.content
                
                # Parse pattern insights
                insights = self._parse_insights_list(response_text)
                
                return insights[:4]  # Limit to 4 insights
                
            except FutureTimeoutError:
                return []
            except Exception as e:
                return []
                
        except Exception as e:
            return []
    
    def generate_learning_points(self, hand_analysis: Dict[str, Any],
                                pattern_insights: List[str] = None) -> List[str]:
        """
        Generate comprehensive learning points.
        
        Args:
            hand_analysis: Hand analysis from StrategyEvaluator
            pattern_insights: Optional pattern recognition insights
        
        Returns:
            list: List of 3-5 learning points
        """
        if not self.api_key_available or not self.client:
            # Fallback to Phase 1's rule-based learning points
            return hand_analysis.get('learning_points', [])
        
        try:
            # Build hand analysis summary
            overall_grade = hand_analysis.get('overall_grade', 'C')
            decisions = hand_analysis.get('decisions', [])
            
            # Summarize decisions
            decision_summary = []
            for decision in decisions[:8]:  # First 8 decisions
                stage = decision.get('stage', 'unknown')
                action = decision.get('action', 'Unknown')
                grade = decision.get('grade', 'C')
                decision_summary.append(f"{stage}: {action} ({grade})")
            
            # Include pattern insights if available
            pattern_text = ""
            if pattern_insights:
                pattern_text = f"\nPattern Insights: {chr(10).join(f'- {insight}' for insight in pattern_insights[:3])}"
            
            # Build prompt
            prompt = f"""Generate comprehensive learning points for this poker hand:

Hand Analysis Summary:
- Overall Grade: {overall_grade}
- Decisions: {chr(10).join(decision_summary) if decision_summary else 'No decisions'}{pattern_text}

Generate 3-5 learning points that include:
- Specific takeaways from this hand
- General strategy principles
- Common mistakes to avoid
- Positive plays to reinforce
- Use consistent terminology (e.g., "continuation bet" not "c-bet")
- Maintain friendly but professional tone
- Focus on learning and improvement

Format as a numbered list of concise learning points."""
            
            # Call LLM with timeout
            try:
                response = self.executor.submit(
                    self._call_openai_api,
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                ).result(timeout=3.0)  # 3 second timeout
                
                response_text = response.choices[0].message.content
                
                # Parse learning points
                learning_points = self._parse_insights_list(response_text)
                
                return learning_points[:5]  # Limit to 5 learning points
                
            except FutureTimeoutError:
                return hand_analysis.get('learning_points', [])
            except Exception as e:
                return hand_analysis.get('learning_points', [])
                
        except Exception as e:
            return hand_analysis.get('learning_points', [])
    
    def _parse_insights_list(self, text: str) -> List[str]:
        """
        Parse numbered or bulleted list from LLM response.
        
        Args:
            text: LLM response text
        
        Returns:
            list: List of insights
        """
        insights = []
        
        # Split by newlines
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering/bullets (1., 2., -, *, etc.)
            line = line.lstrip('0123456789.-* ').strip()
            
            # Remove common prefixes
            prefixes = ['Insight:', 'Learning point:', 'Pattern:', 'Takeaway:']
            for prefix in prefixes:
                if line.lower().startswith(prefix.lower()):
                    line = line[len(prefix):].strip()
                    break
            
            if line and len(line) > 10:  # Minimum length
                insights.append(line)
        
        return insights

