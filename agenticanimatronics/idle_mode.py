import threading
import time
import random
import os
import glob
import pygame


class IdleMode:
    def __init__(self, audio_folder="idle_audio"):
        """
        Idle mode that plays random audio files at random intervals
        
        Args:
            audio_folder: Directory containing audio files for idle mode
        """
        self.audio_folder = audio_folder
        self.is_active = False
        self.idle_thread = None
        self.running = False
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
            self.audio_available = True
        except Exception as e:
            print(f"Warning: Audio initialization failed: {e}")
            self.audio_available = False
        
        # Load available audio files
        self.audio_files = self._load_audio_files()
        
        # Idle timing settings (in seconds)
        self.min_interval = 10  # Minimum time between audio plays
        self.max_interval = 10  # Maximum time between audio plays
    
    def _load_audio_files(self):
        """Load all audio files from the idle audio folder"""
        audio_files = []
        
        if not os.path.exists(self.audio_folder):
            print(f"Creating idle audio folder: {self.audio_folder}")
            os.makedirs(self.audio_folder, exist_ok=True)
            print(f"📁 Add .wav, .mp3, or .ogg files to {self.audio_folder}/ for idle sounds")
            return audio_files
        
        # Support common audio formats
        extensions = ['*.wav', '*.mp3', '*.ogg', '*.m4a']
        
        for ext in extensions:
            files = glob.glob(os.path.join(self.audio_folder, ext))
            audio_files.extend(files)
        
        if audio_files:
            print(f"🎵 Loaded {len(audio_files)} idle audio files")
        else:
            print(f"📁 No audio files found in {self.audio_folder}/")
            print("Add .wav, .mp3, or .ogg files for idle sounds")
            
        return audio_files
    
    def start(self):
        """Start idle mode - begin random audio playback"""
        if not self.audio_available:
            print("🏴‍☠️ Idle mode unavailable - audio system not initialized")
            return
            
        if not self.audio_files:
            print("🏴‍☠️ Idle mode started but no audio files available")
            print(f"Add audio files to {self.audio_folder}/ folder")
        
        self.is_active = True
        self.running = True
        
        # Start the idle audio thread
        self.idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self.idle_thread.start()
        
        print("🏴‍☠️ IDLE MODE ACTIVATED - Pirate will randomly make sounds")
        print("Press 'i' + Enter again to deactivate and restart dialog")
    
    def stop(self):
        """Stop idle mode"""
        self.is_active = False
        self.running = False
        
        # Stop any currently playing audio
        if self.audio_available:
            try:
                pygame.mixer.stop()
            except:
                pass
        
        # Wait for thread to finish
        if self.idle_thread and self.idle_thread.is_alive():
            self.idle_thread.join(timeout=1)
        
        print("🏴‍☠️ IDLE MODE DEACTIVATED")
    
    def _idle_loop(self):
        """Main idle loop that plays random audio at random intervals"""
        while self.running and self.is_active:
            # Wait for a random interval
            interval = random.uniform(self.min_interval, self.max_interval)
            
            # Sleep in small chunks so we can respond to stop requests
            slept = 0
            while slept < interval and self.running and self.is_active:
                time.sleep(0.5)
                slept += 0.5
            
            # If we're still running and have audio files, play one
            if self.running and self.is_active and self.audio_files and self.audio_available:
                self._play_random_audio()
    
    def _play_random_audio(self):
        """Play a random audio file from the collection"""
        if not self.audio_files:
            return
            
        try:
            # Select random audio file
            audio_file = random.choice(self.audio_files)
            
            print(f"🎵 Playing idle sound: {os.path.basename(audio_file)}")
            
            # Load and play the audio
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy() and self.running and self.is_active:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error playing audio file {audio_file}: {e}")
    
    def is_idle_active(self):
        """Check if idle mode is currently active"""
        return self.is_active
    
    def add_audio_file(self, file_path):
        """Add a new audio file to the idle collection"""
        if os.path.exists(file_path):
            self.audio_files.append(file_path)
            print(f"Added audio file: {os.path.basename(file_path)}")
        else:
            print(f"Audio file not found: {file_path}")
    
    def set_timing(self, min_interval=10, max_interval=60):
        """Set the timing intervals for idle audio playback"""
        self.min_interval = min_interval
        self.max_interval = max_interval
        print(f"Idle timing set: {min_interval}-{max_interval} seconds between sounds")