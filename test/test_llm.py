import pytest
import time
from unittest.mock import MagicMock

from agenticanimatronics.llm import LLMHandler


@pytest.fixture
def mock_completion(monkeypatch):
    """Fixture that mocks litellm completion function"""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    
    mock_message.content = "Test LLM response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    mock_completion_func = MagicMock(return_value=mock_response)
    monkeypatch.setattr("agenticanimatronics.llm.completion", mock_completion_func)
    
    return {
        'function': mock_completion_func,
        'response': mock_response,
        'choice': mock_choice,
        'message': mock_message
    }


@pytest.fixture
def mock_encode_image(monkeypatch):
    """Fixture that mocks encode_image function"""
    mock_encode = MagicMock(return_value="base64_encoded_image_data")
    monkeypatch.setattr("agenticanimatronics.llm.encode_image", mock_encode)
    return mock_encode


@pytest.fixture
def mock_time(monkeypatch):
    """Fixture that mocks time.sleep"""
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)
    return mock_sleep


@pytest.fixture
def llm_handler():
    """Fixture that creates an LLMHandler instance"""
    return LLMHandler("test-model")


class TestLLMHandler:
    """Test cases for LLMHandler class"""

    def test_init(self, llm_handler):
        """Test LLMHandler initialization"""
        assert llm_handler.model == "test-model"

    def test_init_default_model(self):
        """Test LLMHandler initialization with default model"""
        handler = LLMHandler()
        assert handler.model == "gemini/gemini-2.5-flash-lite"

    def test_generate_response_success(self, llm_handler, mock_completion):
        """Test successful response generation"""
        prompt = "Test prompt"
        
        result = llm_handler.generate_response(prompt)
        
        assert result == "Test LLM response"
        mock_completion['function'].assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": prompt}]
        )

    @pytest.mark.parametrize("attempt,should_succeed", [
        (1, True),   # Succeeds on first attempt
        (2, True),   # Succeeds on second attempt
        (3, False),  # Fails all attempts
    ])
    def test_generate_response_retry(self, llm_handler, mock_completion, mock_time,
                                    attempt, should_succeed):
        """Test retry logic for response generation"""
        if should_succeed:
            # Fail initial attempts, succeed on the specified attempt
            failures = [Exception("API Error")] * (attempt - 1)
            mock_completion['function'].side_effect = failures + [mock_completion['response']]
        else:
            # Fail all attempts
            mock_completion['function'].side_effect = Exception("API Error")
        
        result = llm_handler.generate_response("Test prompt")
        
        if should_succeed:
            assert result == "Test LLM response"
            assert mock_completion['function'].call_count == attempt
        else:
            assert result == "Arr, me brain be foggy today, matey! Try again later."
            assert mock_completion['function'].call_count == 3
            assert mock_time.call_count == 2  # Sleep between retries

    @pytest.mark.parametrize("prompt", [
        "Simple prompt",
        "",  # Empty prompt
        "Very long prompt with lots of text that goes on and on...",
        "Prompt with special characters: !@#$%^&*()",
    ])
    def test_generate_response_different_prompts(self, llm_handler, mock_completion, prompt):
        """Test response generation with different prompt types"""
        result = llm_handler.generate_response(prompt)
        
        assert result == "Test LLM response"
        call_args = mock_completion['function'].call_args
        assert call_args[1]['messages'][0]['content'] == prompt

    def test_explain_image_success(self, llm_handler, mock_completion, mock_encode_image):
        """Test successful image explanation"""
        image_location = "/path/to/image.jpg"
        prompt = "Describe this image"
        
        result = llm_handler.explain_image(image_location, prompt)
        
        assert result == "Test LLM response"
        mock_encode_image.assert_called_once_with(image_location)
        
        # Verify the message structure
        call_args = mock_completion['function'].call_args
        messages = call_args[1]['messages']
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        
        content = messages[0]['content']
        assert len(content) == 2
        assert content[0]['type'] == 'text'
        assert content[0]['text'] == prompt
        assert content[1]['type'] == 'image_url'
        assert 'base64_encoded_image_data' in content[1]['image_url']['url']

    @pytest.mark.parametrize("attempt,should_succeed", [
        (1, True),   # Succeeds on first attempt
        (2, True),   # Succeeds on second attempt  
        (3, False),  # Fails all attempts
    ])
    def test_explain_image_retry(self, llm_handler, mock_completion, mock_encode_image, 
                                mock_time, attempt, should_succeed):
        """Test retry logic for image explanation"""
        if should_succeed:
            # Fail initial attempts, succeed on the specified attempt
            failures = [Exception("API Error")] * (attempt - 1)
            mock_completion['function'].side_effect = failures + [mock_completion['response']]
        else:
            # Fail all attempts
            mock_completion['function'].side_effect = Exception("API Error")
        
        result = llm_handler.explain_image("/path/to/image.jpg", "Describe image")
        
        if should_succeed:
            assert result == "Test LLM response"
            assert mock_completion['function'].call_count == attempt
        else:
            assert result == "A mysterious stranger stands before me"
            assert mock_completion['function'].call_count == 3
            assert mock_time.call_count == 2  # Sleep between retries

    def test_explain_image_encode_error(self, llm_handler, mock_completion, mock_encode_image, mock_time):
        """Test image explanation when image encoding fails"""
        mock_encode_image.side_effect = Exception("Encode error")
        
        result = llm_handler.explain_image("/path/to/image.jpg", "Describe image")
        
        # Should still attempt to complete but with encoding error
        assert result == "A mysterious stranger stands before me"
        assert mock_encode_image.call_count == 3  # Retries due to encoding error

    @pytest.mark.parametrize("image_path,prompt", [
        ("/path/to/image.jpg", "What do you see?"),
        ("/another/path/image.png", "Describe the person"),
        ("relative/path.gif", ""),  # Empty prompt
    ])
    def test_explain_image_different_inputs(self, llm_handler, mock_completion, 
                                           mock_encode_image, image_path, prompt):
        """Test image explanation with different inputs"""
        result = llm_handler.explain_image(image_path, prompt)
        
        assert result == "Test LLM response"
        mock_encode_image.assert_called_once_with(image_path)
        
        call_args = mock_completion['function'].call_args
        content = call_args[1]['messages'][0]['content']
        assert content[0]['text'] == prompt

    def test_model_parameter_usage(self, mock_completion):
        """Test that custom model parameter is used correctly"""
        custom_model = "gpt-4"
        handler = LLMHandler(custom_model)
        
        handler.generate_response("Test")
        
        call_args = mock_completion['function'].call_args
        assert call_args[1]['model'] == custom_model

    def test_error_messages_different_methods(self, llm_handler, mock_completion, 
                                             mock_encode_image, mock_time):
        """Test that different methods return appropriate error messages"""
        mock_completion['function'].side_effect = Exception("API Error")
        
        # Test generate_response error message
        result1 = llm_handler.generate_response("Test")
        assert "foggy" in result1 and "matey" in result1
        
        # Test explain_image error message  
        result2 = llm_handler.explain_image("/path/image.jpg", "Describe")
        assert "mysterious stranger" in result2

    def test_sleep_timing_between_retries(self, llm_handler, mock_completion, mock_time):
        """Test that sleep is called between retries with correct timing"""
        mock_completion['function'].side_effect = Exception("API Error")
        
        llm_handler.generate_response("Test")
        
        # Should sleep 1 second between each retry
        assert mock_time.call_count == 2
        for call in mock_time.call_args_list:
            assert call[0][0] == 1  # 1 second sleep


class TestLLMHandlerIntegration:
    """Integration-style tests for LLMHandler"""
    
    def test_multiple_requests_different_models(self, mock_completion, mock_encode_image):
        """Test multiple requests with different model configurations"""
        handler1 = LLMHandler("model1")
        handler2 = LLMHandler("model2")
        
        handler1.generate_response("Request 1")
        handler2.generate_response("Request 2")
        
        # Check that correct models were used
        calls = mock_completion['function'].call_args_list
        assert calls[0][1]['model'] == "model1"
        assert calls[1][1]['model'] == "model2"

    def test_mixed_request_types(self, llm_handler, mock_completion, mock_encode_image):
        """Test mixing text and image requests"""
        # Text request
        result1 = llm_handler.generate_response("Text prompt")
        
        # Image request
        result2 = llm_handler.explain_image("/path/image.jpg", "Image prompt")
        
        assert result1 == "Test LLM response"
        assert result2 == "Test LLM response"
        assert mock_completion['function'].call_count == 2
        assert mock_encode_image.call_count == 1

    def test_error_recovery(self, llm_handler, mock_completion):
        """Test that handler recovers from errors and continues working"""
        # First request fails
        mock_completion['function'].side_effect = Exception("Temporary error")
        result1 = llm_handler.generate_response("First request")
        assert "foggy" in result1

        # Second request succeeds
        mock_completion['function'].side_effect = None
        mock_completion['function'].return_value = mock_completion['response']
        result2 = llm_handler.generate_response("Second request")
        assert result2 == "Test LLM response"