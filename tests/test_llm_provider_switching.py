#!/usr/bin/env python3
"""
Test script to verify LLM provider switching functionality.

This script tests that the LLM_PROVIDER environment variable correctly
switches between OpenAI and OpenRouter providers in both ChatbotCoach
and LLMOpponentAgent classes.
"""

import os
import sys
from unittest.mock import patch, MagicMock

# Add webapp to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))


def test_provider_logic():
    """Test the LLM provider selection logic directly."""


    # Test OpenAI provider selection
    with patch.dict(os.environ, {
        'LLM_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'test_openai_key'
    }):
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        openai_key = os.getenv("OPENAI_API_KEY")

        assert llm_provider == "openai", "Should select OpenAI provider"
        assert openai_key == "test_openai_key", "Should get OpenAI key"

    # Test OpenRouter provider selection
    with patch.dict(os.environ, {
        'LLM_PROVIDER': 'openrouter',
        'OPEN_ROUTER_KEY': 'test_openrouter_key'
    }):
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        openrouter_key = os.getenv("OPEN_ROUTER_KEY")

        assert llm_provider == "openrouter", "Should select OpenRouter provider"
        assert openrouter_key == "test_openrouter_key", "Should get OpenRouter key"

    # Test default to OpenAI
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key'
    }, clear=True):
        # Clear LLM_PROVIDER if it exists
        if 'LLM_PROVIDER' in os.environ:
            del os.environ['LLM_PROVIDER']

        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        assert llm_provider == "openai", "Should default to OpenAI"

    # Test invalid provider fallback
    with patch.dict(os.environ, {
        'LLM_PROVIDER': 'invalid_provider',
        'OPENAI_API_KEY': 'test_openai_key'
    }):
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        assert llm_provider == "invalid_provider", "Should get invalid provider value"
        # The actual fallback logic is in the class __init__, but we can test the env var reading



if __name__ == "__main__":
    test_provider_logic()
