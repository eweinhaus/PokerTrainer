#!/usr/bin/env python3
"""Test RLCard availability"""

try:
    import rlcard
    print("✅ RLCard available")
    print(f"RLCard version: {rlcard.__version__ if hasattr(rlcard, '__version__') else 'unknown'}")

    # Test basic import
    from rlcard.games.nolimitholdem import NolimitholdemGame
    print("✅ NolimitholdemGame import successful")

except ImportError as e:
    print(f"❌ RLCard not available: {e}")

try:
    import flask
    print("✅ Flask available")
except ImportError as e:
    print(f"❌ Flask not available: {e}")

try:
    import openai
    print("✅ OpenAI available")
except ImportError as e:
    print(f"❌ OpenAI not available: {e}")
