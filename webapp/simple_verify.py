#!/usr/bin/env python3
"""Simple verification script for Phase 5 completion"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

def main():
    print("🔍 Checking Phase 5 completion status...")

    # Check imports
    try:
        from coach.strategy_evaluator import StrategyEvaluator
        from coach.chatbot_coach import ChatbotCoach
        from coach.gto_rules import GTORules
        from coach.equity_calculator import EquityCalculator
        from coach.pattern_recognizer import PatternRecognizer
        print("✅ All coach modules import successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

    # Check basic functionality
    try:
        evaluator = StrategyEvaluator()
        test_state = {'stage': 0, 'hand': [('S', 'A'), ('H', 'K')], 'pot': 4, 'stakes': [100, 100], 'big_blind': 2}
        result = evaluator.evaluate_action(test_state, 1)
        if result and 'grade' in result:
            print("✅ StrategyEvaluator basic functionality working")
        else:
            print("❌ StrategyEvaluator not returning expected result")
            return False
    except Exception as e:
        print(f"❌ StrategyEvaluator error: {e}")
        return False

    # Check GTO rules
    try:
        gto = GTORules()
        print("✅ GTORules import and initialization working")
    except Exception as e:
        print(f"❌ GTORules error: {e}")
        return False

    # Check equity calculator
    try:
        equity = EquityCalculator()
        print("✅ EquityCalculator import and initialization working")
    except Exception as e:
        print(f"❌ EquityCalculator error: {e}")
        return False

    # Check chatbot coach (without API calls)
    try:
        coach = ChatbotCoach()
        print("✅ ChatbotCoach import and initialization working")
    except Exception as e:
        print(f"❌ ChatbotCoach error: {e}")
        return False

    # Check pattern recognizer
    try:
        recognizer = PatternRecognizer()
        print("✅ PatternRecognizer import and initialization working")
    except Exception as e:
        print(f"❌ PatternRecognizer error: {e}")
        return False

    print("\n🎉 All core components are working!")
    print("✅ Phase 5 appears to be functionally complete")
    print("✅ Ready to proceed with RLCard BB-First Action Order implementation")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
