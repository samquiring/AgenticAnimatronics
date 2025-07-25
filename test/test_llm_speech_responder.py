import pytest
import time
from unittest.mock import MagicMock

from agenticanimatronics.llm_speech_responder import LLMSpeechResponder


@pytest.fixture
def mock_llm_handler(monkeypatch):
    """Fixture that mocks LLMHandler"""
    mock_handler = MagicMock()
    mock_handler.generate_response.return_value = "Test LLM response"
    monkeypatch.setattr("agenticanimatronics.llm_speech_responder.LLMHandler", 
                       lambda: mock_handler)
    return mock_handler


@pytest.fixture
def mock_pirate_chatbot(monkeypatch):
    """Fixture that mocks PirateChatBot"""
    mock_chatbot = MagicMock()
    mock_chatbot.forward.return_value = "Arr, test response matey!"
    monkeypatch.setattr("agenticanimatronics.llm_speech_responder.PirateChatBot", 
                       lambda: mock_chatbot)
    return mock_chatbot


@pytest.fixture
def mock_elevenlabs(monkeypatch):
    """Fixture that mocks ElevenLabs client and related functions"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_client.generate.return_value = mock_response
    
    mock_stream_func = MagicMock()
    mock_stream_func.return_value = b"audio_data"
    
    mock_voice_settings = MagicMock()
    
    monkeypatch.setattr("agenticanimatronics.llm_speech_responder.ElevenLabs", 
                       lambda **kwargs: mock_client)
    monkeypatch.setattr("agenticanimatronics.llm_speech_responder.stream", 
                       mock_stream_func)
    monkeypatch.setattr("agenticanimatronics.llm_speech_responder.VoiceSettings", 
                       mock_voice_settings)
    
    return {
        'client': mock_client,
        'stream': mock_stream_func,
        'voice_settings': mock_voice_settings,
        'response': mock_response
    }


@pytest.fixture
def mock_pyaudio(monkeypatch):
    """Fixture that mocks PyAudio"""
    mock_audio = MagicMock()
    monkeypatch.setattr("pyaudio.PyAudio", lambda: mock_audio)
    return mock_audio


@pytest.fixture
def mock_time(monkeypatch):
    """Fixture that mocks time functions"""
    mock_time_module = MagicMock()
    mock_time_module.time.side_effect = [0.0, 1.0]  # For timing measurements
    mock_time_module.sleep.return_value = None
    monkeypatch.setattr("time.time", mock_time_module.time)
    monkeypatch.setattr("time.sleep", mock_time_module.sleep)
    return mock_time_module


@pytest.fixture
def llm_speech_responder(mock_llm_handler, mock_pirate_chatbot, mock_elevenlabs, mock_pyaudio):
    """Fixture that creates LLMSpeechResponder with mocked dependencies"""
    return LLMSpeechResponder("test_voice_id")


class TestLLMSpeechResponder:
    """Test cases for LLMSpeechResponder class"""

    def test_init_success(self, llm_speech_responder, mock_elevenlabs):
        """Test successful initialization"""
        assert llm_speech_responder.eleven_labs_voice_id == "test_voice_id"
        assert llm_speech_responder.conversation_history == []
        assert llm_speech_responder.interrupt_lock is False

    def test_text_to_speech_stream_success(self, llm_speech_responder, mock_elevenlabs):
        """Test successful text-to-speech conversion"""
        result = llm_speech_responder.text_to_speech_stream("Test text")
        
        assert result == b"audio_data"
        mock_elevenlabs['client'].generate.assert_called_once()
        mock_elevenlabs['stream'].assert_called_once_with(mock_elevenlabs['response'])

    @pytest.mark.parametrize("attempt,should_succeed", [
        (1, True),   # First attempt succeeds
        (3, False),  # All attempts fail
    ])
    def test_text_to_speech_stream_retry(self, llm_speech_responder, mock_elevenlabs, 
                                        mock_time, attempt, should_succeed):
        """Test text-to-speech retry logic"""
        if should_succeed:
            # Fail first attempts, succeed on last
            mock_elevenlabs['client'].generate.side_effect = [
                Exception("API Error")] * (attempt - 1) + [mock_elevenlabs['response']]
        else:
            # Fail all attempts
            mock_elevenlabs['client'].generate.side_effect = Exception("API Error")
        
        result = llm_speech_responder.text_to_speech_stream("Test text")
        
        if should_succeed:
            assert result == b"audio_data"
            assert mock_elevenlabs['client'].generate.call_count == attempt
        else:
            assert result is None
            assert mock_elevenlabs['client'].generate.call_count == 3
            assert mock_time.sleep.call_count == 2  # Sleeps between retries

    @pytest.mark.parametrize("user_response,assistant_response", [
        ("Hello", "Hi there"),
        ("", "Empty response"),
        ("Long user input", "Long assistant response"),
    ])
    def test_update_conversational_history(self, llm_speech_responder, 
                                          user_response, assistant_response):
        """Test conversation history updates"""
        initial_length = len(llm_speech_responder.conversation_history)
        
        llm_speech_responder.update_conversational_history(user_response, assistant_response)
        
        assert len(llm_speech_responder.conversation_history) == initial_length + 2
        assert llm_speech_responder.conversation_history[-2] == {
            "role": "user", "content": user_response
        }
        assert llm_speech_responder.conversation_history[-1] == {
            "role": "assistant", "content": assistant_response
        }

    def test_generate_success(self, llm_speech_responder, mock_pirate_chatbot, 
                             mock_elevenlabs, mock_time):
        """Test successful generation of response"""
        user_description = "A person in a red shirt"
        user_response = "Hello pirate"
        
        llm_speech_responder.generate(user_description, user_response)
        
        mock_pirate_chatbot.forward.assert_called_once_with(
            history=[{'role': 'user', 'content': 'Hello pirate'}, {'role': 'assistant', 'content': 'Arr, test response matey!'}],
            user_prompt=user_response,
            user_description=user_description
        )
        mock_elevenlabs['client'].generate.assert_called_once()
        
        # Check conversation history was updated
        assert len(llm_speech_responder.conversation_history) == 2
        assert llm_speech_responder.conversation_history[0]["content"] == user_response
        assert llm_speech_responder.conversation_history[1]["content"] == "Arr, test response matey!"

    def test_generate_chatbot_error(self, llm_speech_responder, mock_pirate_chatbot, mock_time):
        """Test generation when chatbot fails"""
        mock_pirate_chatbot.forward.side_effect = Exception("Chatbot error")
        
        llm_speech_responder.generate("user desc", "user input")
        
        # Should add fallback response to history
        assert len(llm_speech_responder.conversation_history) == 2
        assert llm_speech_responder.conversation_history[1]["content"] == \
               "Arr, something went wrong with me voice, matey!"

    def test_generate_speech_error(self, llm_speech_responder, mock_pirate_chatbot, 
                                  mock_elevenlabs, mock_time):
        """Test generation when speech synthesis fails"""
        mock_elevenlabs['client'].generate.side_effect = Exception("Speech error")
        
        llm_speech_responder.generate("user desc", "user input")
        
        # Should still update conversation history with chatbot response
        assert len(llm_speech_responder.conversation_history) == 2
        assert llm_speech_responder.conversation_history[1]["content"] == "Arr, test response matey!"

    def test_generate_with_existing_history(self, llm_speech_responder, mock_pirate_chatbot):
        """Test generation with existing conversation history"""
        # Add some existing history
        llm_speech_responder.conversation_history = [
            {"role": "user", "content": "Previous user message"},
            {"role": "assistant", "content": "Previous assistant message"}
        ]
        
        llm_speech_responder.generate("user desc", "new message")
        
        # Should pass existing history to chatbot
        mock_pirate_chatbot.forward.assert_called_once()
        call_args = mock_pirate_chatbot.forward.call_args
        assert len(call_args.kwargs['history']) == 4
        assert call_args.kwargs['history'][0]["content"] == "Previous user message"

    @pytest.mark.parametrize("user_description,expected_description", [
        ("A tall person", "A tall person"),
        ("", ""),
        (None, None),
    ])
    def test_generate_user_description_handling(self, llm_speech_responder, mock_pirate_chatbot,
                                               user_description, expected_description):
        """Test handling of different user description values"""
        llm_speech_responder.generate(user_description, "test message")
        
        mock_pirate_chatbot.forward.assert_called_once()
        call_args = mock_pirate_chatbot.forward.call_args
        assert call_args.kwargs['user_description'] == expected_description

    def test_timing_measurement(self, llm_speech_responder, mock_pirate_chatbot, mock_time):
        """Test that response timing is measured"""
        llm_speech_responder.generate("user desc", "user input")
        
        # time() should be called twice for timing measurement
        assert mock_time.time.call_count == 2


class TestLLMSpeechResponderIntegration:
    """Integration-style tests for LLMSpeechResponder"""
    
    def test_full_conversation_flow(self, llm_speech_responder, mock_pirate_chatbot, mock_elevenlabs):
        """Test a full conversation flow"""
        # First exchange
        llm_speech_responder.generate("Person in red", "Hello")
        
        # Second exchange
        mock_pirate_chatbot.forward.return_value = "Arr, how be ye doing?"
        llm_speech_responder.generate("Person in red", "I'm doing well")
        
        # Check conversation history has both exchanges
        assert len(llm_speech_responder.conversation_history) == 4
        assert llm_speech_responder.conversation_history[0]["content"] == "Hello"
        assert llm_speech_responder.conversation_history[2]["content"] == "I'm doing well"
        
        # Check that second call received the history
        second_call_args = mock_pirate_chatbot.forward.call_args
        assert len(second_call_args.kwargs['history']) == 4

    def test_error_recovery(self, llm_speech_responder, mock_pirate_chatbot, mock_elevenlabs):
        """Test that system recovers from errors and continues working"""
        # First call fails
        mock_pirate_chatbot.forward.side_effect = Exception("First error")
        llm_speech_responder.generate("user desc", "first message")
        
        # Second call succeeds
        mock_pirate_chatbot.forward.side_effect = None
        mock_pirate_chatbot.forward.return_value = "Arr, working again!"
        llm_speech_responder.generate("user desc", "second message")
        
        # Should have fallback response from first call and normal response from second
        assert len(llm_speech_responder.conversation_history) == 4
        assert "something went wrong" in llm_speech_responder.conversation_history[1]["content"]
        assert llm_speech_responder.conversation_history[3]["content"] == "Arr, working again!"