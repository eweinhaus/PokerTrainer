"""
Tests for ChatbotCoach class
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
from coach.chatbot_coach import ChatbotCoach


class TestChatbotCoach(unittest.TestCase):
    """Test ChatbotCoach functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock OpenAI client
        self.mock_client = Mock()
        self.mock_response = Mock()
        self.mock_response.choices = [Mock()]
        self.mock_response.choices[0].message = Mock()
        self.mock_response.choices[0].message.content = "Test response from coach"
        
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_chatbot_coach_initialization(self, mock_openai):
        """Test ChatbotCoach can be instantiated"""
        mock_openai.return_value = self.mock_client
        
        coach = ChatbotCoach()
        
        self.assertIsNotNone(coach)
        self.assertIsNotNone(coach.system_prompt)
        # Updated to match new system prompt content
        self.assertIn("poker coach", coach.system_prompt.lower())
        self.assertIn("gto", coach.system_prompt.lower())
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_system_prompt_contains_required_elements(self, mock_openai):
        """Test system prompt contains all required elements"""
        mock_openai.return_value = self.mock_client
        
        coach = ChatbotCoach()
        prompt = coach.system_prompt.lower()
        
        # Check for required elements
        self.assertIn("expert poker coach", prompt)
        self.assertIn("gto", prompt)
        self.assertIn("no limit", prompt)
        self.assertIn("heads-up", prompt)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_chat_without_api_key(self, mock_openai):
        """Test chat returns fallback when API key is missing"""
        # Don't set API key
        with patch.dict(os.environ, {}, clear=True):
            coach = ChatbotCoach()
            response = coach.chat("What are pot odds?", None, "test_session")
            
            self.assertIn('response', response)
            self.assertIn('timestamp', response)
            # Should be fallback response
            self.assertIn('unavailable', response['response'].lower() or 'tip', response['response'].lower())
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_conversation_history_management(self, mock_openai):
        """Test conversation history is maintained"""
        mock_openai.return_value = self.mock_client
        
        coach = ChatbotCoach()
        session_id = "test_session"
        
        # Initially empty
        history = coach._get_conversation_history(session_id)
        self.assertEqual(len(history), 0)
        
        # Add messages
        coach._add_to_history(session_id, "user", "Hello")
        coach._add_to_history(session_id, "assistant", "Hi there")
        
        # Check history
        history = coach._get_conversation_history(session_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['role'], 'user')
        self.assertEqual(history[0]['content'], 'Hello')
        self.assertEqual(history[1]['role'], 'assistant')
        self.assertEqual(history[1]['content'], 'Hi there')
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_conversation_history_limit(self, mock_openai):
        """Test conversation history is limited to last 10 messages"""
        mock_openai.return_value = self.mock_client
        
        coach = ChatbotCoach()
        session_id = "test_session"
        
        # Add 15 messages
        for i in range(15):
            coach._add_to_history(session_id, "user", f"Message {i}")
        
        # Should only have last 10
        history = coach._get_conversation_history(session_id)
        self.assertEqual(len(history), 10)
        self.assertEqual(history[0]['content'], 'Message 5')  # First of last 10
        self.assertEqual(history[-1]['content'], 'Message 14')  # Last message
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_context_injection(self, mock_openai):
        """Test context injection works correctly"""
        mock_openai.return_value = self.mock_client
        
        coach = ChatbotCoach()
        
        # Test with game context
        game_context = {
            'hand': [0, 1],  # Ace of spades, 2 of spades
            'public_cards': [2, 3, 4],
            'pot': 20,
            'big_blind': 2,
            'all_chips': [100, 100],
            'button_id': 0,
            'current_player': 0
        }
        
        context_text = coach._inject_context(game_context, "What should I do now?")
        
        self.assertIsNotNone(context_text)
        self.assertIn("Current Hand", context_text)
        self.assertIn("Board", context_text)
        self.assertIn("Pot", context_text)
        self.assertIn("Position", context_text)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_context_injection_relevance(self, mock_openai):
        """Test context is only injected when relevant"""
        mock_openai.return_value = self.mock_client
        
        coach = ChatbotCoach()
        
        game_context = {
            'hand': [0, 1],
            'public_cards': [2, 3, 4],
            'pot': 20,
            'big_blind': 2,
            'all_chips': [100, 100]
        }
        
        # Relevant question - should include context
        context1 = coach._inject_context(game_context, "What should I do now?")
        self.assertIsNotNone(context1)
        
        # Not relevant question - should not include context
        context2 = coach._inject_context(game_context, "What are pot odds?")
        self.assertIsNone(context2)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_format_response(self, mock_openai):
        """Test response formatting"""
        mock_openai.return_value = self.mock_client
        
        coach = ChatbotCoach()
        
        response = coach._format_response("Test response")
        
        self.assertIn('response', response)
        self.assertIn('timestamp', response)
        self.assertEqual(response['response'], 'Test response')
        self.assertIsNotNone(response['timestamp'])
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('coach.chatbot_coach.OpenAI')
    def test_clear_history(self, mock_openai):
        """Test history clearing"""
        mock_openai.return_value = self.mock_client
        
        coach = ChatbotCoach()
        session_id = "test_session"
        
        # Add some history
        coach._add_to_history(session_id, "user", "Hello")
        self.assertEqual(len(coach._get_conversation_history(session_id)), 1)
        
        # Clear history
        coach.clear_history(session_id)
        self.assertEqual(len(coach._get_conversation_history(session_id)), 0)


if __name__ == '__main__':
    unittest.main()


