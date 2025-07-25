import pytest
from array import array
from unittest.mock import MagicMock

from agenticanimatronics.mutable_microphone_stream import MutableMicrophoneStream


@pytest.fixture
def mock_pyaudio_module(monkeypatch):
    """Fixture that mocks the entire pyaudio module"""
    mock_pyaudio_class = MagicMock()
    mock_stream = MagicMock()
    mock_instance = MagicMock()
    
    # Mock device info
    mock_instance.get_default_input_device_info.return_value = {"name": "Test Microphone"}
    
    # Mock stream creation
    mock_instance.open.return_value = mock_stream
    mock_stream.is_active.return_value = True
    mock_stream.start_stream.return_value = None
    mock_stream.stop_stream.return_value = None
    mock_stream.close.return_value = None
    mock_stream.read.return_value = b'\x00\x01' * 50  # Mock audio data
    
    # Mock PyAudio class
    mock_pyaudio_class.return_value = mock_instance
    mock_pyaudio_class.paInt16 = "paInt16"
    
    monkeypatch.setattr("pyaudio.PyAudio", mock_pyaudio_class)
    
    return {
        'class': mock_pyaudio_class,
        'instance': mock_instance,
        'stream': mock_stream
    }


@pytest.fixture
def mock_array(monkeypatch):
    """Fixture that mocks array operations"""
    def mock_array_func(type_code, data):
        mock_arr = MagicMock()
        # Simulate max() behavior
        if data:
            mock_arr.__iter__ = lambda: iter([100, 200, 300])  # Mock audio levels
            mock_arr.__bool__ = lambda: True
        else:
            mock_arr.__iter__ = lambda: iter([])
            mock_arr.__bool__ = lambda: False
        return mock_arr
    
    monkeypatch.setattr("array.array", mock_array_func)
    return mock_array_func


@pytest.fixture
def microphone_stream(mock_pyaudio_module, mock_array):
    """Fixture that creates a MutableMicrophoneStream with mocked dependencies"""
    return MutableMicrophoneStream(sample_rate=16000, threshold=500)


class TestMutableMicrophoneStream:
    """Test cases for MutableMicrophoneStream class"""

    def test_init_success(self, microphone_stream, mock_pyaudio_module):
        """Test successful initialization"""
        assert microphone_stream.sample_rate == 16000
        assert microphone_stream.is_muted is False
        assert microphone_stream.threshold == 500
        assert microphone_stream._open is True
        
        # Verify PyAudio was set up correctly
        mock_pyaudio_module['instance'].open.assert_called_once()
        mock_pyaudio_module['stream'].start_stream.assert_called_once()

    @pytest.mark.parametrize("sample_rate,device_index,threshold", [
        (44100, None, 1000),
        (22050, 1, 250),
        (16000, 0, 100),
    ])
    def test_init_different_parameters(self, mock_pyaudio_module, mock_array, 
                                      sample_rate, device_index, threshold):
        """Test initialization with different parameters"""
        stream = MutableMicrophoneStream(
            sample_rate=sample_rate,
            device_index=device_index, 
            threshold=threshold
        )
        
        assert stream.sample_rate == sample_rate
        assert stream.threshold == threshold
        
        # Check that device_index was passed to pyaudio.open
        call_args = mock_pyaudio_module['instance'].open.call_args
        assert call_args[1]['input_device_index'] == device_index

    def test_iterator_protocol(self, microphone_stream):
        """Test that MutableMicrophoneStream implements iterator protocol"""
        assert iter(microphone_stream) == microphone_stream
        assert hasattr(microphone_stream, '__next__')

    @pytest.mark.parametrize("is_muted,threshold,audio_level,expected_silence", [
        (False, 500, 400, True),   # Unmuted, below threshold - should return silence
        (True, 500, 600, True),    # Muted, above threshold - should return silence
        (True, 500, 400, True),    # Muted, below threshold - should return silence
    ])
    def test_next_audio_processing(self, microphone_stream, mock_pyaudio_module, mock_array,
                                  is_muted, threshold, audio_level, expected_silence):
        """Test audio processing logic in __next__"""
        microphone_stream.is_muted = is_muted
        microphone_stream.threshold = threshold
        
        # Mock the audio level by controlling what max() returns
        def mock_array_with_level(type_code, data):
            mock_arr = MagicMock()
            mock_arr.__iter__ = lambda: iter([audio_level])
            mock_arr.__bool__ = lambda: True
            return mock_arr
        
        import builtins
        original_max = builtins.max
        builtins.max = lambda arr: audio_level if arr else 0
        
        try:
            result = next(microphone_stream)
            
            if expected_silence:
                assert result == b'\x00' * (microphone_stream._chunk_size * 2)
            else:
                assert result == b'\x00\x01' * 50  # Mock audio data
        finally:
            builtins.max = original_max

    def test_next_stream_read_error(self, microphone_stream, mock_pyaudio_module):
        """Test handling of stream read errors"""
        mock_pyaudio_module['stream'].read.side_effect = Exception("Read error")
        
        result = next(microphone_stream)
        
        # Should return silence on error
        assert result == b'\x00' * (microphone_stream._chunk_size * 2)

    def test_next_keyboard_interrupt(self, microphone_stream, mock_pyaudio_module):
        """Test handling of KeyboardInterrupt"""
        mock_pyaudio_module['stream'].read.side_effect = KeyboardInterrupt()
        
        with pytest.raises(StopIteration):
            next(microphone_stream)

    def test_next_when_closed(self, microphone_stream):
        """Test __next__ when stream is closed"""
        microphone_stream._open = False
        
        with pytest.raises(StopIteration):
            next(microphone_stream)

    def test_close(self, microphone_stream, mock_pyaudio_module):
        """Test closing the stream"""
        microphone_stream.close()
        
        assert microphone_stream._open is False
        mock_pyaudio_module['stream'].stop_stream.assert_called_once()
        mock_pyaudio_module['stream'].close.assert_called_once()
        mock_pyaudio_module['instance'].terminate.assert_called_once()

    def test_close_inactive_stream(self, microphone_stream, mock_pyaudio_module):
        """Test closing when stream is not active"""
        mock_pyaudio_module['stream'].is_active.return_value = False
        
        microphone_stream.close()
        
        assert microphone_stream._open is False
        mock_pyaudio_module['stream'].stop_stream.assert_not_called()
        mock_pyaudio_module['stream'].close.assert_called_once()

    def test_close_missing_attributes(self, microphone_stream):
        """Test closing when some attributes are missing"""
        # Remove some attributes to simulate initialization failure
        delattr(microphone_stream, '_stream')
        
        # Should not raise an exception
        microphone_stream.close()
        assert microphone_stream._open is False

    @pytest.mark.parametrize("initial_muted,expected_muted", [
        (False, True),   # Mute when unmuted
        (True, False),   # Unmute when muted
    ])
    def test_toggle_mute(self, microphone_stream, initial_muted, expected_muted):
        """Test mute toggle functionality"""
        microphone_stream.is_muted = initial_muted
        
        microphone_stream.toggle_mute()
        
        assert microphone_stream.is_muted == expected_muted

    def test_mute(self, microphone_stream):
        """Test mute functionality"""
        microphone_stream.mute()
        assert microphone_stream.is_muted is True

    def test_unmute(self, microphone_stream):
        """Test unmute functionality"""
        microphone_stream.is_muted = True
        microphone_stream.unmute()
        assert microphone_stream.is_muted is False

    def test_context_manager(self, mock_pyaudio_module, mock_array):
        """Test context manager functionality"""
        with MutableMicrophoneStream(sample_rate=16000) as stream:
            assert stream._open is True
            assert isinstance(stream, MutableMicrophoneStream)
        
        # Should be closed after context
        assert stream._open is False
        mock_pyaudio_module['stream'].close.assert_called()

    def test_data_padding(self, microphone_stream, mock_pyaudio_module):
        """Test data padding when insufficient data is received"""
        # Mock insufficient data
        insufficient_data = b'\x00' * 50  # Less than expected
        mock_pyaudio_module['stream'].read.return_value = insufficient_data
        
        result = next(microphone_stream)
        
        # Should pad with zeros to expected length
        expected_length = microphone_stream._chunk_size * 2
        assert len(result) == expected_length

    def test_exception_on_overflow_false(self, microphone_stream, mock_pyaudio_module):
        """Test that exception_on_overflow=False is used"""
        next(microphone_stream)
        
        # Verify that read was called with exception_on_overflow=False
        call_args = mock_pyaudio_module['stream'].read.call_args
        assert call_args[1]['exception_on_overflow'] is False

    def test_chunk_size_calculation(self, mock_pyaudio_module, mock_array):
        """Test chunk size calculation"""
        sample_rate = 44100
        stream = MutableMicrophoneStream(sample_rate=sample_rate)
        
        expected_chunk_size = int(sample_rate * 0.05)  # 50ms
        assert stream._chunk_size == expected_chunk_size


class TestMutableMicrophoneStreamIntegration:
    """Integration-style tests for MutableMicrophoneStream"""
    
    def test_audio_processing_flow(self, microphone_stream, mock_pyaudio_module):
        """Test the complete audio processing flow"""
        # Simulate receiving multiple audio chunks
        audio_data = [b'\x00\x01' * 50, b'\x01\x02' * 50, b'\x02\x03' * 50]
        mock_pyaudio_module['stream'].read.side_effect = audio_data
        
        results = []
        for _ in range(3):
            results.append(next(microphone_stream))
        
        # All should return audio data (assuming above threshold)
        assert len(results) == 3
        assert all(len(result) == microphone_stream._chunk_size * 2 for result in results)

    def test_error_recovery(self, microphone_stream, mock_pyaudio_module):
        """Test that stream recovers from temporary errors"""
        # First call fails, second succeeds
        mock_pyaudio_module['stream'].read.side_effect = [
            Exception("Temporary error"),
            b'\x11\x11' * 50
        ]
        
        # First call should return silence due to error
        result1 = next(microphone_stream)
        silence = b'\x00' * (microphone_stream._chunk_size * 2)
        assert result1 == silence
        
        # Second call should succeed
        result2 = next(microphone_stream)
        assert result2 != silence