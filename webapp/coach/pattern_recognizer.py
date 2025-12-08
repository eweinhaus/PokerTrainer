"""
Pattern Recognizer

Analyzes patterns across multiple hands in a session to identify
recurring mistakes, good habits, tendencies, and improvement areas.
"""

# logging removed
from typing import List, Dict, Any, Optional
from collections import defaultdict



class PatternRecognizer:
    """
    Recognizes patterns across multiple hands in a session.
    
    Identifies:
    - Consistent mistakes
    - Good habits
    - Player tendencies
    - Improvement areas
    """
    
    def __init__(self):
        """Initialize pattern recognizer with caching."""
        # Cache for pattern analysis results: {session_id: {hand_count: analysis}}
        self.pattern_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
    
    def get_recent_hands(self, session_id: str, hand_history_storage: Dict[str, List], count: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve last N hands from session history.
        
        Args:
            session_id: Session identifier
            hand_history_storage: Hand history storage from app.py
            count: Number of hands to retrieve (default: 10)
        
        Returns:
            list: List of hand analysis dictionaries (last N hands)
        """
        try:
            # Get all decisions for this session
            all_decisions = hand_history_storage.get(session_id, [])
            
            if not all_decisions:
                return []
            
            # Group decisions by hand (simplified: assume hands are separated by game over)
            # For now, return last N*10 decisions (assuming ~10 decisions per hand)
            # In a full implementation, we'd track hand boundaries
            recent_decisions = all_decisions[-count*10:] if len(all_decisions) > count*10 else all_decisions
            
            # For MVP, return decision summaries grouped by approximate hands
            # This is simplified - in production, we'd track actual hand boundaries
            hands = []
            current_hand = []
            
            for decision in recent_decisions:
                if decision.get('player_id') == 0:  # Only player decisions
                    current_hand.append({
                        'stage': decision.get('stage', 0),
                        'action': decision.get('action', 'Unknown'),
                        'pot': decision.get('pot', 0)
                    })
            
            # Group into approximate hands (every 5-10 decisions = 1 hand)
            hand_size = 8  # Approximate decisions per hand
            for i in range(0, len(current_hand), hand_size):
                hand_decisions = current_hand[i:i+hand_size]
                if hand_decisions:
                    hands.append({
                        'decisions': hand_decisions,
                        'decision_count': len(hand_decisions)
                    })
            
            # Return last N hands
            return hands[-count:] if len(hands) > count else hands
            
        except Exception as e:
            return []
    
    def extract_decision_patterns(self, hand_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract decision patterns from multiple hands.
        
        Args:
            hand_list: List of hand dictionaries with decisions
        
        Returns:
            dict: {
                'decision_summaries': list,  # Summary of each hand's decisions
                'action_frequencies': dict,  # Frequency of each action type
                'stage_distribution': dict,   # Distribution of decisions by stage
                'common_actions': list        # Most common actions
            }
        """
        if not hand_list:
            return {
                'decision_summaries': [],
                'action_frequencies': {},
                'stage_distribution': {},
                'common_actions': []
            }
        
        action_frequencies = defaultdict(int)
        stage_distribution = defaultdict(int)
        decision_summaries = []
        
        for hand in hand_list:
            decisions = hand.get('decisions', [])
            hand_summary = {
                'decision_count': len(decisions),
                'actions': [],
                'stages': []
            }
            
            for decision in decisions:
                action = decision.get('action', 'Unknown')
                stage = decision.get('stage', 0)
                
                action_frequencies[action] += 1
                stage_distribution[stage] += 1
                
                hand_summary['actions'].append(action)
                hand_summary['stages'].append(stage)
            
            decision_summaries.append(hand_summary)
        
        # Find most common actions
        common_actions = sorted(
            action_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'decision_summaries': decision_summaries,
            'action_frequencies': dict(action_frequencies),
            'stage_distribution': dict(stage_distribution),
            'common_actions': [action for action, count in common_actions]
        }
    
    def identify_patterns(self, hand_list: List[Dict[str, Any]], 
                         current_hand_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Identify recurring patterns across hands.
        
        Args:
            hand_list: List of hand dictionaries
            current_hand_analysis: Optional current hand analysis for comparison
        
        Returns:
            dict: {
                'consistent_mistakes': list,
                'good_habits': list,
                'tendencies': list,
                'improvement_areas': list
            }
        """
        if not hand_list or len(hand_list) < 2:
            return {
                'consistent_mistakes': [],
                'good_habits': [],
                'tendencies': [],
                'improvement_areas': []
            }
        
        # Extract decision patterns
        patterns = self.extract_decision_patterns(hand_list)
        
        # Analyze for patterns
        consistent_mistakes = []
        good_habits = []
        tendencies = []
        improvement_areas = []
        
        # Analyze action frequencies
        action_freq = patterns['action_frequencies']
        total_decisions = sum(action_freq.values())
        
        if total_decisions > 0:
            # Identify tendencies (actions that appear frequently)
            for action, count in action_freq.items():
                frequency = count / total_decisions
                if frequency > 0.4:  # Appears in >40% of decisions
                    if 'fold' in action.lower():
                        tendencies.append(f"Tendency to fold frequently ({frequency*100:.0f}% of decisions)")
                    elif 'raise' in action.lower():
                        tendencies.append(f"Tendency to raise frequently ({frequency*100:.0f}% of decisions)")
                    elif 'call' in action.lower():
                        tendencies.append(f"Tendency to call frequently ({frequency*100:.0f}% of decisions)")
        
        # Analyze stage distribution
        stage_dist = patterns['stage_distribution']
        if stage_dist:
            preflop_count = stage_dist.get(0, 0)
            postflop_count = sum(stage_dist.get(i, 0) for i in [1, 2, 3])
            total_stages = preflop_count + postflop_count
            
            if total_stages > 0:
                preflop_pct = preflop_count / total_stages
                if preflop_pct > 0.7:
                    tendencies.append("Most decisions made preflop (hands end early)")
                elif preflop_pct < 0.3:
                    tendencies.append("Most decisions made postflop (hands go to later streets)")
        
        # If we have current hand analysis, compare patterns
        if current_hand_analysis:
            current_decisions = current_hand_analysis.get('decisions', [])
            if current_decisions:
                # Check if current hand follows established patterns
                current_actions = [d.get('action', '') for d in current_decisions]
                for action in current_actions:
                    if action in action_freq:
                        # This action has been seen before
                        if 'fold' in action.lower() and action_freq[action] / total_decisions > 0.3:
                            improvement_areas.append(f"Consider more aggressive play - folding too frequently")
        
        return {
            'consistent_mistakes': consistent_mistakes,
            'good_habits': good_habits,
            'tendencies': tendencies,
            'improvement_areas': improvement_areas,
            'pattern_data': patterns
        }
    
    def get_cached_analysis(self, session_id: str, hand_count: int) -> Optional[Dict[str, Any]]:
        """
        Get cached pattern analysis if available.
        
        Args:
            session_id: Session identifier
            hand_count: Number of hands analyzed
        
        Returns:
            dict: Cached analysis or None
        """
        if session_id in self.pattern_cache:
            return self.pattern_cache[session_id].get(hand_count)
        return None
    
    def cache_analysis(self, session_id: str, hand_count: int, analysis: Dict[str, Any]):
        """
        Cache pattern analysis result.
        
        Args:
            session_id: Session identifier
            hand_count: Number of hands analyzed
            analysis: Analysis result to cache
        """
        if session_id not in self.pattern_cache:
            self.pattern_cache[session_id] = {}
        
        self.pattern_cache[session_id][hand_count] = analysis
        
        # Limit cache size (keep last 10 analyses per session)
        if len(self.pattern_cache[session_id]) > 10:
            oldest_key = min(self.pattern_cache[session_id].keys())
            del self.pattern_cache[session_id][oldest_key]



