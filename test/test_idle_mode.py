import pytest
import os
import glob
import threading
from unittest.mock import MagicMock

from agenticanimatronics.idle_mode import IdleMode


@pytest.fixture
def mock_pygame(monkeypatch):
    """Fixture that mocks pygame functionality"""
    mock_pygame_module = MagicMock()
    mock_mixer = MagicMock()
    mock_music = MagicMock()
    
    mock_mixer.init.return_value = None
    mock_mixer.stop.return_value = None
    mock_music.load.return_value = None
    mock_music.play.return_value = None
    mock_music.get_busy.return_value = False
    
    mock_mixer.music = mock_music
    mock_pygame_module.mixer = mock_mixer
    
    monkeypatch.setattr("pygame.mixer", mock_mixer)
    return {
        'pygame': mock_pygame_module,
        'mixer': mock_mixer,
        'music': mock_music
    }


@pytest.fixture
def mock_file_system(monkeypatch):
    """Fixture that mocks file system operations"""
    mock_os = MagicMock()
    mock_glob = MagicMock()
    
    # Mock os.path.exists to return True by default
    mock_os.path.exists.return_value = True
    mock_os.makedirs.return_value = None
    mock_os.path.basename.side_effect = lambda path: path.split('/')[-1]
    mock_os.path.join.side_effect = lambda *args: '/'.join(args)
    
    # Mock glob.glob to return some test files
    mock_glob.glob.side_effect = [
        ['idle_audio/test1.wav'],
        ['idle_audio/test2.mp3'],
        ['idle_audio/test3.ogg'],
        []
    ]
    
    monkeypatch.setattr("os.path.exists", mock_os.path.exists)
    monkeypatch.setattr("os.makedirs", mock_os.makedirs)
    monkeypatch.setattr("os.path.basename", mock_os.path.basename)
    monkeypatch.setattr("os.path.join", mock_os.path.join)
    monkeypatch.setattr("glob.glob", mock_glob.glob)
    
    return {
        'os': mock_os,
        'glob': mock_glob
    }


@pytest.fixture
def mock_threading(monkeypatch, mock_thread):
    """Fixture that mocks threading"""
    monkeypatch.setattr("threading.Thread", lambda **kwargs: mock_thread)
    return mock_thread


@pytest.fixture
def idle_mode(mock_pygame, mock_file_system):
    """Fixture that creates an IdleMode with mocked dependencies"""
    return IdleMode("test_audio_folder")


class TestIdleMode:
    """Test cases for IdleMode class"""

    def test_init_success(self, idle_mode, mock_pygame, mock_file_system):
        """Test successful initialization of IdleMode"""
        assert idle_mode.audio_folder == "test_audio_folder"
        assert idle_mode.is_active is False
        assert idle_mode.running is False
        assert idle_mode.audio_available is True
        assert idle_mode.min_interval == 10
        assert idle_mode.max_interval == 10
        assert len(idle_mode.audio_files) == 3  # From mock_file_system

    def test_init_pygame_failure(self, monkeypatch, mock_file_system):
        """Test initialization when pygame fails"""
        def failing_pygame_init():
            raise Exception("Pygame init failed")
        
        mock_mixer = MagicMock()
        mock_mixer.init = failing_pygame_init
        monkeypatch.setattr("pygame.mixer", mock_mixer)
        
        idle_mode = IdleMode("test_audio_folder")
        
        assert idle_mode.audio_available is False

    @pytest.mark.parametrize("folder_exists,expected_files", [
        (True, 3),    # Folder exists with files
        (False, 0),   # Folder doesn't exist
    ])
    def test_load_audio_files(self, monkeypatch, mock_pygame, folder_exists, expected_files):
        """Test loading audio files with different folder states"""
        mock_os = MagicMock()
        mock_glob = MagicMock()
        
        mock_os.path.exists.return_value = folder_exists
        mock_os.makedirs.return_value = None
        mock_os.path.join.side_effect = lambda *args: '/'.join(args)
        
        if folder_exists:
            mock_glob.glob.side_effect = [['test1.wav'], ['test2.mp3'], ['test3.ogg'], []]
        else:
            mock_glob.glob.return_value = []
        
        monkeypatch.setattr("os.path.exists", mock_os.path.exists)
        monkeypatch.setattr("os.makedirs", mock_os.makedirs)
        monkeypatch.setattr("os.path.join", mock_os.path.join)
        monkeypatch.setattr("glob.glob", mock_glob.glob)
        
        idle_mode = IdleMode("test_folder")
        
        assert len(idle_mode.audio_files) == expected_files
        if not folder_exists:
            mock_os.makedirs.assert_called_once_with("test_folder", exist_ok=True)

    @pytest.mark.parametrize("audio_available,audio_files,should_start", [
        (True, ['test.wav'], True),     # Normal case
        (False, ['test.wav'], False),   # No audio system
        (True, [], True),               # No audio files but system available
    ])
    def test_start(self, idle_mode, mock_threading, mock_pygame, 
                   audio_available, audio_files, should_start):
        """Test starting idle mode under different conditions"""
        idle_mode.audio_available = audio_available
        idle_mode.audio_files = audio_files
        
        idle_mode.start()
        
        if should_start and audio_available:
            assert idle_mode.is_active is True
            assert idle_mode.running is True
            mock_threading.start.assert_called_once()
        elif not audio_available:
            assert idle_mode.is_active is False

    def test_stop(self, idle_mode, mock_pygame, mock_thread):
        """Test stopping idle mode"""
        idle_mode.is_active = True
        idle_mode.running = True
        idle_mode.idle_thread = mock_thread
        mock_thread.is_alive.return_value = True
        
        idle_mode.stop()
        
        assert idle_mode.is_active is False
        assert idle_mode.running is False
        mock_pygame['mixer'].stop.assert_called_once()
        mock_thread.join.assert_called_once_with(timeout=1)

    def test_stop_pygame_error(self, idle_mode, mock_thread):
        """Test stopping idle mode when pygame fails"""
        idle_mode.is_active = True
        idle_mode.running = True
        idle_mode.idle_thread = mock_thread
        
        # Mock pygame to raise exception
        mock_mixer = MagicMock()
        mock_mixer.stop.side_effect = Exception("Pygame error")
        idle_mode.audio_available = True
        
        # Should not raise exception
        idle_mode.stop()
        
        assert idle_mode.is_active is False
        assert idle_mode.running is False

    def test_is_idle_active(self, idle_mode):
        """Test idle active status check"""
        assert idle_mode.is_idle_active() is False
        
        idle_mode.is_active = True
        assert idle_mode.is_idle_active() is True

    @pytest.mark.parametrize("has_files,should_play", [
        (True, True),    # Has files - should play
        (False, False),  # No files - should not play
    ])
    def test_play_random_audio(self, idle_mode, mock_pygame, mock_file_system,
                              has_files, should_play, monkeypatch):
        """Test playing random audio files"""
        if has_files:
            idle_mode.audio_files = ['test1.wav', 'test2.mp3']
        else:
            idle_mode.audio_files = []
        
        # Mock random.choice
        mock_random = MagicMock()
        mock_random.choice.return_value = 'test1.wav'
        monkeypatch.setattr("random.choice", mock_random.choice)
        
        idle_mode._play_random_audio()
        
        if should_play:
            mock_pygame['music'].load.assert_called_once_with('test1.wav')
            mock_pygame['music'].play.assert_called_once()
            mock_random.choice.assert_called_once_with(['test1.wav', 'test2.mp3'])
        else:
            mock_pygame['music'].load.assert_not_called()
            mock_pygame['music'].play.assert_not_called()

    def test_play_random_audio_error(self, idle_mode, mock_pygame, monkeypatch):
        """Test playing audio when pygame raises exception"""
        idle_mode.audio_files = ['test.wav']
        
        # Mock pygame to raise exception
        mock_pygame['music'].load.side_effect = Exception("Audio load error")
        
        # Mock random.choice
        mock_random = MagicMock()
        mock_random.choice.return_value = 'test.wav'
        monkeypatch.setattr("random.choice", mock_random.choice)
        
        # Should not raise exception
        idle_mode._play_random_audio()
        
        mock_pygame['music'].load.assert_called_once_with('test.wav')

    def test_idle_loop_stops_when_not_running(self, idle_mode, monkeypatch):
        """Test that idle loop stops when running is False"""
        idle_mode.running = True
        idle_mode.is_active = True
        
        # Mock time.sleep to avoid actual waiting
        mock_time = MagicMock()
        monkeypatch.setattr("time.sleep", mock_time.sleep)
        
        # Mock _play_random_audio
        idle_mode._play_random_audio = MagicMock()
        
        # Simulate stopping after first iteration
        def stop_after_first_call(*args):
            idle_mode.running = False
        
        mock_time.sleep.side_effect = stop_after_first_call
        
        idle_mode._idle_loop()
        
        # Should have tried to sleep at least once
        mock_time.sleep.assert_called()

    def test_idle_loop_stops_when_not_active(self, idle_mode, monkeypatch):
        """Test that idle loop stops when is_active is False"""
        idle_mode.running = True
        idle_mode.is_active = False
        
        # Mock time.sleep and _play_random_audio
        mock_time = MagicMock()
        monkeypatch.setattr("time.sleep", mock_time.sleep)
        idle_mode._play_random_audio = MagicMock()
        
        idle_mode._idle_loop()
        
        # Should not have called sleep or play_random_audio
        mock_time.sleep.assert_not_called()
        idle_mode._play_random_audio.assert_not_called()