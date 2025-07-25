import pytest
import multiprocessing
import threading
from unittest.mock import MagicMock


@pytest.fixture
def mock_queue():
    """Mock multiprocessing queue"""
    queue = MagicMock()
    queue.get.return_value = "Test user description"
    return queue


@pytest.fixture
def mock_process():
    """Mock multiprocessing process"""
    process = MagicMock()
    process.is_alive.return_value = False
    process.start.return_value = None
    process.terminate.return_value = None
    process.join.return_value = None
    return process


@pytest.fixture
def mock_thread():
    """Mock threading thread"""
    thread = MagicMock()
    thread.is_alive.return_value = False
    thread.start.return_value = None
    thread.join.return_value = None
    return thread


@pytest.fixture
def mock_pyaudio():
    """Mock PyAudio instance"""
    pyaudio_mock = MagicMock()
    pyaudio_mock.get_default_input_device_info.return_value = {"name": "Test Device"}
    
    # Mock audio stream
    stream_mock = MagicMock()
    stream_mock.read.return_value = b'\x00' * 100  # Silent audio data
    stream_mock.is_active.return_value = True
    pyaudio_mock.open.return_value = stream_mock
    
    return pyaudio_mock


@pytest.fixture
def mock_assemblyai_transcriber():
    """Mock AssemblyAI transcriber"""
    transcriber = MagicMock()
    transcriber.connect.return_value = None
    transcriber.stream.return_value = None
    transcriber.close.return_value = None
    return transcriber


@pytest.fixture
def mock_image_analysis():
    """Mock ImageAnalysis instance"""
    image_analysis = MagicMock()
    image_analysis.take_and_analyse_image.return_value = "Test user description"
    return image_analysis


@pytest.fixture
def mock_llm_speech_responder():
    """Mock LLMSpeechResponder instance"""
    responder = MagicMock()
    responder.generate.return_value = None
    responder.conversation_history = []
    return responder


@pytest.fixture
def mock_microphone_stream():
    """Mock MutableMicrophoneStream"""
    stream = MagicMock()
    stream.mute.return_value = None
    stream.unmute.return_value = None
    stream.close.return_value = None
    return stream


@pytest.fixture
def mock_idle_mode():
    """Mock IdleMode instance"""
    idle_mode = MagicMock()
    idle_mode.start.return_value = None
    idle_mode.stop.return_value = None
    idle_mode.is_idle_active.return_value = False
    return idle_mode