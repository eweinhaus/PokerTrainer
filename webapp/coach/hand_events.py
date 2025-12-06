"""
Hand Event Schema - Single Source of Truth for Action History

This module defines the consistent event schema used throughout the system
for representing all poker hand events (blinds, actions, community cards, wins).
"""

from dataclasses import dataclass, field
from typing import Optional, List
import time


@dataclass
class HandEvent:
    """
    Represents a single event in a poker hand.
    
    This is the single source of truth for action history and action labels.
    All events are stored in chronological order and used by both backend
    and frontend for consistent display.
    """
    # Event metadata
    t: float = field(default_factory=time.time)  # Monotonic timestamp
    stage: str = 'preflop'  # 'preflop' | 'flop' | 'turn' | 'river' | 'showdown'
    kind: str = 'action'  # 'blind' | 'action' | 'community' | 'win'
    
    # Player information (None for community cards)
    player_idx: Optional[int] = None  # 0-1 for heads-up, None for board events
    
    # Financial information
    amount: Optional[float] = None  # Chips associated with event (bet, blind, win)
    pot: float = 0.0  # Pot size AFTER this event
    
    # Card information
    cards: List[str] = field(default_factory=list)  # Used for community cards
    
    # Display information
    label: str = ''  # Human-readable label (e.g., "Raise to 3 BB", "Post SB", "Flop")
    
    # Additional context
    action_value: Optional[int] = None  # RLCard action value (0-4) for action events
    bet_amount: Optional[float] = None  # Bet amount for action events
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            't': self.t,
            'stage': self.stage,
            'kind': self.kind,
            'player_idx': self.player_idx,
            'amount': self.amount,
            'pot': self.pot,
            'cards': self.cards,
            'label': self.label,
            'action_value': self.action_value,
            'bet_amount': self.bet_amount
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create HandEvent from dictionary"""
        return cls(
            t=data.get('t', time.time()),
            stage=data.get('stage', 'preflop'),
            kind=data.get('kind', 'action'),
            player_idx=data.get('player_idx'),
            amount=data.get('amount'),
            pot=data.get('pot', 0.0),
            cards=data.get('cards', []),
            label=data.get('label', ''),
            action_value=data.get('action_value'),
            bet_amount=data.get('bet_amount')
        )

