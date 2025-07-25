import multiprocessing
import threading
import pyaudio
import assemblyai as aai

from agenticanimatronics.discord_handler import dual_discord_sink
from agenticanimatronics.image_analysis import ImageAnalysis
from agenticanimatronics.initializers import assembly_ai_key, logs_webhook, alerts_webhook
from agenticanimatronics.mutable_microphone_stream import MutableMicrophoneStream
from agenticanimatronics.llm_speech_responder import LLMSpeechResponder
from agenticanimatronics.idle_mode import IdleMode
from loguru import logger

aai.settings.api_key = assembly_ai_key


class PirateAgent:
    def __init__(
            self,
            eleven_labs_voice_id="Myn1LuZgd2qPMOg9BNtC",
            image_prompt="Describe the person/people in this image"
    ):
        try:
            self.transcriber = aai.RealtimeTranscriber(
                sample_rate=16000,
                on_data=self.on_data,
                on_error=self.on_error,
                on_open=self.on_open,
                on_close=self.on_close,
                end_utterance_silence_threshold=700,
                disable_partial_transcripts=True
            )
        except Exception:
            logger.exception("Error initializing transcriber")
            raise
            
        try:
            self.microphone_stream = MutableMicrophoneStream(sample_rate=16000)
        except Exception:
            logger.exception("Error initializing microphone")
            logger.error("Make sure your microphone is connected and permissions are granted")
            raise
            
        self.user_transcript = []
        self.pirate_agent_thread = None
        
        # Initialize audio playback components
        try:
            self.audio_player = pyaudio.PyAudio()
        except Exception:
            logger.exception("Error initializing audio player")
            self.audio_player = None
            
        self.skipped_pirate_audio = True  # this ensures the last audio from the pirate isn't picked up
        
        try:
            self.pirate_agent = LLMSpeechResponder(eleven_labs_voice_id=eleven_labs_voice_id)
        except Exception:
            logger.exception("Error initializing speech responder")
            raise
            
        self.image_analysis = ImageAnalysis(prompt=image_prompt)
        self.image_analysis_thread = None
        self.queue = multiprocessing.Queue()
        self.user_description = ""
        
        # Pause/resume controls
        self.is_paused = False
        self.keyboard_thread = None
        self.running = True
        
        # Idle mode
        self.idle_mode = IdleMode()
        self.in_idle_mode = False
        
        # Photo update system
        self.photo_update_thread = None
        self.photo_update_interval = 30  # Update photo every 30 seconds
        self.photo_update_running = False

    @staticmethod
    def on_open(session_opened: aai.RealtimeSessionOpened):
        logger.info("Session ID:", session_opened.session_id)

    def on_data(self, transcript: aai.RealtimeTranscript):
        if self.is_paused or self.in_idle_mode:
            return
            
        # Start photo update system if not already running
        if not self.photo_update_running:
            self.start_photo_updates()
            
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            # Create a thread for pirate response
            if not self.pirate_agent_thread or not self.pirate_agent_thread.is_alive():
                self.pirate_agent_thread = threading.Thread(
                    target=self.pirate_agent.generate,
                    args=(self.user_description, transcript.text),  # Function arguments
                    daemon=True  # Optional: makes thread exit when main program exits
                )
                self.pirate_agent_thread.start()
            else:
                logger.debug("Pirate is still speaking")
            logger.info(f"User said: {transcript.text}")
        else:
            # For partial transcripts
            logger.info(transcript.text, end="\r")

    @staticmethod
    def on_error(error: aai.RealtimeError):
        logger.error("An error occurred:", error)

    @staticmethod
    def on_close():
        logger.info("Closing Session")

    def monitor_keyboard(self):
        """Monitor keyboard input for pause/resume commands"""
        logger.info("🏴‍☠️ Type 'p' + Enter to pause/resume, 'i' + Enter for idle mode, "
                    "'r' + Enter to restart, 'q' + Enter to quit"
                    )
        
        while self.running:
            try:
                command = input().strip().lower()
                if command == 'p':
                    self.toggle_pause()
                elif command == 'i':
                    self.toggle_idle_mode()
                elif command == 'r':
                    self.restart_dialog()
                elif command == 'q':
                    logger.info("\nQuitting...")
                    self.running = False
                    break
                elif command == 'help':
                    logger.info("Commands: 'p' = pause/resume, 'i' = idle mode, 'r' = restart, 'q' = quit")
            except (EOFError, KeyboardInterrupt):
                logger.info("\nQuitting...")
                self.running = False
                break
            except Exception:
                logger.exception("Input error")
                continue

    def toggle_pause(self):
        """Toggle pause/resume state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            logger.info("\n🏴‍☠️ Pirate PAUSED - Press 'p' to resume, 'q' to quit")
            self.microphone_stream.mute()
        else:
            logger.info("\n🏴‍☠️ Pirate RESUMED - Press 'p' to pause, 'q' to quit")
            self.microphone_stream.unmute()

    def toggle_idle_mode(self):
        """Toggle idle mode on/off"""
        if self.in_idle_mode:
            self.deactivate_idle_mode()
        else:
            self.activate_idle_mode()

    def activate_idle_mode(self):
        """Activate idle mode - pirate will randomly play audio files"""
        self.in_idle_mode = True
        self.is_paused = True  # Pause normal operation
        self.microphone_stream.mute()
        self.idle_mode.start()

    def deactivate_idle_mode(self):
        """Deactivate idle mode and return to normal operation"""
        self.in_idle_mode = False
        self.idle_mode.stop()
        self.is_paused = False  # Resume normal operation
        self.microphone_stream.unmute()
        # Restart dialog to begin fresh conversation
        self.restart_dialog()

    def start_photo_updates(self):
        """Start the periodic photo update system"""
        if self.photo_update_running:
            return
            
        self.photo_update_running = True
        self.photo_update_thread = threading.Thread(target=self._photo_update_loop, daemon=True)
        self.photo_update_thread.start()
        logger.info(f"📸 Started photo updates every {self.photo_update_interval} seconds")

    def stop_photo_updates(self):
        """Stop the periodic photo update system"""
        self.photo_update_running = False
        if self.photo_update_thread and self.photo_update_thread.is_alive():
            self.photo_update_thread.join(timeout=2)
        logger.info("📸 Stopped photo updates")

    def _photo_update_loop(self):
        """Background loop that regularly takes photos and updates user description"""
        import time
        
        # Take initial photo
        self._update_user_photo()
        
        while self.photo_update_running and self.running:
            # Sleep in small chunks so we can respond to stop requests
            slept = 0
            while slept < self.photo_update_interval and self.photo_update_running and self.running:
                time.sleep(1)
                slept += 1
            
            # Take another photo if still running and not in idle mode
            if self.photo_update_running and self.running and not self.in_idle_mode:
                self._update_user_photo()

    def _update_user_photo(self):
        """Take a photo and update the user description"""
        try:
            logger.debug("📸 Taking updated photo for user description")
            
            # Create a new queue for this photo update
            photo_queue = multiprocessing.Queue()
            
            # Start image analysis process
            photo_process = multiprocessing.Process(
                target=self.image_analysis.take_and_analyse_image,
                args=("", photo_queue),
                daemon=True
            )
            photo_process.start()
            
            # Wait for result with timeout
            try:
                new_description = photo_queue.get(timeout=10)
                self.user_description = new_description
                logger.info(f"📸 Updated user description: {new_description[:50]}...")
            except Exception as e:
                logger.warning(f"📸 Photo update failed {e} - keeping previous description")
            
            # Clean up process
            if photo_process.is_alive():
                photo_process.terminate()
                photo_process.join(timeout=1)
                
        except Exception:
            logger.exception("Error updating user photo")

    def set_photo_update_interval(self, seconds):
        """Set the interval for photo updates"""
        self.photo_update_interval = max(10, seconds)  # Minimum 10 seconds
        logger.info(f"📸 Photo update interval set to {self.photo_update_interval} seconds")

    def restart_dialog(self):
        logger.info("🏴‍☠️ Restarting Pirate Dialog - Starting fresh conversation")
        self.user_description = ""
        self.image_analysis_thread = None
        self.queue = multiprocessing.Queue()
        self.user_transcript = []
        # Clear conversation history in the speech responder
        if hasattr(self.pirate_agent, 'conversation_history'):
            self.pirate_agent.conversation_history = []

    def transcribe(self):
        """
        Start transcribing audio from the microphone for a specified duration.
        """
        try:
            logger.info("🏴‍☠️ Starting Pirate Agent... Press 'p' to pause/resume, 'q' to quit, "
                        "'i' + Enter for idle mode, 'r' + Enter to restart"
                        )
            
            # Start keyboard monitoring thread
            self.keyboard_thread = threading.Thread(target=self.monitor_keyboard, daemon=True)
            self.keyboard_thread.start()
            
            self.transcriber.connect()

            # Start streaming audio from the microphone
            self.transcriber.stream(self.microphone_stream)

            return ' '.join(self.user_transcript)
        except KeyboardInterrupt:
            logger.exception("\nShutting down gracefully...")
            self.running = False
        except Exception:
            logger.exception("Error in transcribe")
            self.running = False

    def cleanup(self):
        """
        Clean up resources when done.
        """
        logger.info("Cleaning up resources...")
        self.running = False
        
        # Stop transcriber
        try:
            if hasattr(self, 'transcriber'):
                self.transcriber.close()
        except Exception as e:
            logger.info(f"Error closing transcriber: {e}")
        
        # Close microphone stream
        try:
            if hasattr(self, 'microphone_stream'):
                self.microphone_stream.close()
        except Exception:
            logger.exception("Error closing microphone")
            
        # Stop idle mode if active
        try:
            if self.in_idle_mode:
                self.idle_mode.stop()
        except Exception:
            logger.exception("Error stopping idle mode")
            
        # Stop photo updates
        try:
            self.stop_photo_updates()
        except Exception:
            logger.exception("Error stopping photo updates")
            
        # Terminate audio player
        try:
            if hasattr(self, 'audio_player') and self.audio_player:
                self.audio_player.terminate()
        except Exception:
            logger.exception("Error terminating audio player")
            
        # Join threads
        try:
            if self.pirate_agent_thread and self.pirate_agent_thread.is_alive():
                self.pirate_agent_thread.join(timeout=2)
        except Exception:
            logger.exception("Error joining pirate thread")
            
        try:
            if self.image_analysis_thread and self.image_analysis_thread.is_alive():
                self.image_analysis_thread.terminate()
        except Exception:
            logger.exception("Error terminating image analysis")


# Example usage
def run_pirate_agent():
    logger.add(
        dual_discord_sink(logs_webhook, alerts_webhook),
        level="INFO"
    )
    agent = PirateAgent()
    try:
        result = agent.transcribe()
        
        # Keep running until user quits
        while agent.running:
            try:
                import time
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
                
        logger.info(f"Final transcript: {result}")
    except Exception:
        logger.exception("Error running pirate agent")
    finally:
        agent.cleanup()


if __name__ == "__main__":
    run_pirate_agent()
