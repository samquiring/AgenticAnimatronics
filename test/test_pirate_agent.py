import pytest
import multiprocessing
import threading
from unittest.mock import MagicMock
import assemblyai as aai

from agenticanimatronics.pirate_agent import PirateAgent


@pytest.fixture
def mock_all_dependencies(monkeypatch, mock_pyaudio, mock_microphone_stream,
                         mock_assemblyai_transcriber, mock_image_analysis,
                         mock_llm_speech_responder, mock_idle_mode, mock_queue):
    """Fixture that patches all PirateAgent dependencies"""
    monkeypatch.setattr("pyaudio.PyAudio", lambda: mock_pyaudio)
    monkeypatch.setattr("agenticanimatronics.pirate_agent.MutableMicrophoneStream", 
                       lambda **kwargs: mock_microphone_stream)
    monkeypatch.setattr("assemblyai.RealtimeTranscriber", lambda **kwargs: mock_assemblyai_transcriber)
    monkeypatch.setattr("agenticanimatronics.pirate_agent.ImageAnalysis", 
                       lambda **kwargs: mock_image_analysis)
    monkeypatch.setattr("agenticanimatronics.pirate_agent.LLMSpeechResponder", 
                       lambda **kwargs: mock_llm_speech_responder)
    monkeypatch.setattr("agenticanimatronics.pirate_agent.IdleMode", lambda: mock_idle_mode)
    monkeypatch.setattr("multiprocessing.Queue", lambda: mock_queue)
    
    return {
        'pyaudio': mock_pyaudio,
        'microphone_stream': mock_microphone_stream,
        'transcriber': mock_assemblyai_transcriber,
        'image_analysis': mock_image_analysis,
        'llm_speech_responder': mock_llm_speech_responder,
        'idle_mode': mock_idle_mode,
        'queue': mock_queue
    }


@pytest.fixture
def mock_threading(monkeypatch, mock_thread):
    """Fixture that patches threading"""
    monkeypatch.setattr("threading.Thread", lambda **kwargs: mock_thread)
    return mock_thread


@pytest.fixture
def pirate_agent(mock_all_dependencies):
    """Fixture that creates a PirateAgent with all dependencies mocked"""
    return PirateAgent()


class TestPirateAgent:
    """Test cases for PirateAgent class"""

    def test_init_success(self, pirate_agent):
        """Test successful initialization of PirateAgent"""
        assert pirate_agent.is_paused is False
        assert pirate_agent.running is True
        assert pirate_agent.in_idle_mode is False
        assert pirate_agent.photo_update_interval == 30
        assert pirate_agent.photo_update_running is False

    def test_init_microphone_error(self, monkeypatch, mock_pyaudio, mock_assemblyai_transcriber):
        """Test initialization failure when microphone fails"""
        monkeypatch.setattr("pyaudio.PyAudio", lambda: mock_pyaudio)
        monkeypatch.setattr("assemblyai.RealtimeTranscriber", lambda **kwargs: mock_assemblyai_transcriber)
        
        def failing_microphone(**kwargs):
            raise Exception("Microphone not available")
        
        monkeypatch.setattr("agenticanimatronics.pirate_agent.MutableMicrophoneStream", 
                           failing_microphone)
        
        with pytest.raises(Exception, match="Microphone not available"):
            PirateAgent()

    @pytest.mark.parametrize("initial_state,expected_paused,expected_mute_calls,expected_unmute_calls", [
        (False, True, 1, 0),  # First toggle should pause
        (True, False, 0, 1),  # Second toggle should resume
    ])
    def test_toggle_pause(self, pirate_agent, mock_all_dependencies, 
                         initial_state, expected_paused, expected_mute_calls, expected_unmute_calls):
        """Test pause/resume functionality"""
        pirate_agent.is_paused = initial_state
        mock_all_dependencies['microphone_stream'].reset_mock()
        
        pirate_agent.toggle_pause()
        
        assert pirate_agent.is_paused == expected_paused
        assert mock_all_dependencies['microphone_stream'].mute.call_count == expected_mute_calls
        assert mock_all_dependencies['microphone_stream'].unmute.call_count == expected_unmute_calls

    @pytest.mark.parametrize("initial_idle,expected_idle,expected_paused", [
        (False, True, True),   # Activate idle mode
        (True, False, False),  # Deactivate idle mode
    ])
    def test_toggle_idle_mode(self, pirate_agent, mock_all_dependencies, 
                             initial_idle, expected_idle, expected_paused):
        """Test idle mode toggle functionality"""
        pirate_agent.in_idle_mode = initial_idle
        pirate_agent.is_paused = initial_idle
        mock_all_dependencies['idle_mode'].reset_mock()
        mock_all_dependencies['microphone_stream'].reset_mock()
        
        # Mock _update_user_photo for deactivation path
        pirate_agent._update_user_photo = MagicMock()
        
        pirate_agent.toggle_idle_mode()
        
        assert pirate_agent.in_idle_mode == expected_idle
        assert pirate_agent.is_paused == expected_paused
        
        if expected_idle:  # Activating
            mock_all_dependencies['idle_mode'].start.assert_called_once()
            mock_all_dependencies['microphone_stream'].mute.assert_called_once()
        else:  # Deactivating
            mock_all_dependencies['idle_mode'].stop.assert_called_once()
            mock_all_dependencies['microphone_stream'].unmute.assert_called_once()

    @pytest.mark.parametrize("is_paused,in_idle_mode,should_process", [
        (True, False, False),   # Paused - should not process
        (False, True, False),   # In idle mode - should not process
        (True, True, False),    # Both paused and idle - should not process
        (False, False, True),   # Normal state - should process
    ])
    def test_on_data_early_return(self, pirate_agent, is_paused, in_idle_mode, should_process):
        """Test that on_data returns early when paused or in idle mode"""
        pirate_agent.is_paused = is_paused
        pirate_agent.in_idle_mode = in_idle_mode
        pirate_agent.start_photo_updates = MagicMock()
        
        transcript = MagicMock()
        transcript.text = "Hello"
        
        result = pirate_agent.on_data(transcript)
        
        if should_process:
            pirate_agent.start_photo_updates.assert_called_once()
        else:
            pirate_agent.start_photo_updates.assert_not_called()
            assert result is None

    @pytest.mark.parametrize("interval,expected_interval", [
        (60, 60),    # Valid interval
        (5, 10),     # Below minimum - should be clamped
        (0, 10),     # Zero - should be clamped
        (-5, 10),    # Negative - should be clamped
    ])
    def test_set_photo_update_interval(self, pirate_agent, interval, expected_interval):
        """Test setting photo update interval with various values"""
        pirate_agent.set_photo_update_interval(interval)
        assert pirate_agent.photo_update_interval == expected_interval

    def test_start_photo_updates(self, pirate_agent, mock_threading):
        """Test starting photo update system"""
        assert pirate_agent.photo_update_running is False
        
        pirate_agent.start_photo_updates()
        
        assert pirate_agent.photo_update_running is True
        mock_threading.start.assert_called_once()
        
        # Test that calling again doesn't start another thread
        mock_threading.reset_mock()
        pirate_agent.start_photo_updates()
        mock_threading.start.assert_not_called()

    def test_stop_photo_updates(self, pirate_agent, mock_thread):
        """Test stopping photo update system"""
        pirate_agent.photo_update_running = True
        pirate_agent.photo_update_thread = mock_thread
        mock_thread.is_alive.return_value = True
        
        pirate_agent.stop_photo_updates()
        
        assert pirate_agent.photo_update_running is False
        mock_thread.join.assert_called_once_with(timeout=2)

    def test_restart_dialog(self, pirate_agent, mock_all_dependencies):
        """Test dialog restart functionality"""
        # Set up initial state
        pirate_agent.user_description = "old description"
        pirate_agent.image_analysis_thread = MagicMock()
        pirate_agent.user_transcript = ["old", "transcript"]
        pirate_agent.pirate_agent.conversation_history = ["old", "history"]
        
        pirate_agent.restart_dialog()
        
        assert pirate_agent.user_description == ""
        assert pirate_agent.image_analysis_thread is None
        assert pirate_agent.user_transcript == []
        assert pirate_agent.pirate_agent.conversation_history == []

    def test_cleanup(self, pirate_agent, mock_all_dependencies, mock_thread):
        """Test cleanup functionality"""
        pirate_agent.in_idle_mode = True
        pirate_agent.pirate_agent_thread = mock_thread
        pirate_agent.stop_photo_updates = MagicMock()
        
        pirate_agent.cleanup()
        
        assert pirate_agent.running is False
        mock_all_dependencies['transcriber'].close.assert_called_once()
        mock_all_dependencies['microphone_stream'].close.assert_called_once()
        mock_all_dependencies['idle_mode'].stop.assert_called_once()
        mock_all_dependencies['pyaudio'].terminate.assert_called_once()
        pirate_agent.stop_photo_updates.assert_called_once()

    def test_update_user_photo_success(self, pirate_agent, mock_all_dependencies, monkeypatch, mock_process):
        """Test successful photo update"""
        # Mock multiprocessing components
        mock_photo_queue = MagicMock()
        mock_photo_queue.get.return_value = "New user description"
        monkeypatch.setattr("multiprocessing.Queue", lambda: mock_photo_queue)
        monkeypatch.setattr("multiprocessing.Process", lambda **kwargs: mock_process)
        
        pirate_agent._update_user_photo()
        
        assert pirate_agent.user_description == "New user description"
        mock_process.start.assert_called_once()

    def test_update_user_photo_timeout(self, pirate_agent, mock_all_dependencies, monkeypatch, mock_process):
        """Test photo update with timeout"""
        # Mock multiprocessing components
        mock_photo_queue = MagicMock()
        mock_photo_queue.get.side_effect = Exception("Timeout")
        monkeypatch.setattr("multiprocessing.Queue", lambda: mock_photo_queue)
        monkeypatch.setattr("multiprocessing.Process", lambda **kwargs: mock_process)
        
        original_description = "Original description"
        pirate_agent.user_description = original_description
        
        pirate_agent._update_user_photo()
        
        # Should keep original description on timeout
        assert pirate_agent.user_description == original_description
        mock_process.start.assert_called_once()


class TestPirateAgentStaticMethods:
    """Test static methods of PirateAgent"""

    def test_on_open(self):
        """Test on_open static method"""
        session_mock = MagicMock()
        session_mock.session_id = "test-session-123"
        
        # Should not raise any exceptions
        PirateAgent.on_open(session_mock)

    def test_on_error(self):
        """Test on_error static method"""
        error_mock = MagicMock()
        error_mock.__str__ = lambda: "Test error"
        
        # Should not raise any exceptions
        PirateAgent.on_error(error_mock)

    def test_on_close(self):
        """Test on_close static method"""
        # Should not raise any exceptions
        PirateAgent.on_close()