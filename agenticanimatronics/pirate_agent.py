import multiprocessing
import threading
import pyaudio
import assemblyai as aai

from agenticanimatronics.image_analysis import ImageAnalysis
from agenticanimatronics.initializers import assembly_ai_key
from agenticanimatronics.mutable_microphone_stream import MutableMicrophoneStream
from agenticanimatronics.llm_speech_responder import LLMSpeechResponder

aai.settings.api_key = assembly_ai_key


class PirateAgent:
    def __init__(self, eleven_labs_voice_id="Myn1LuZgd2qPMOg9BNtC", image_prompt="Describe the person/people in this image"):
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
        except Exception as e:
            print(f"Error initializing transcriber: {e}")
            raise
            
        try:
            self.microphone_stream = MutableMicrophoneStream(sample_rate=16000)
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            print("Make sure your microphone is connected and permissions are granted")
            raise
            
        self.user_transcript = []
        self.pirate_agent_thread = None
        
        # Initialize audio playback components
        try:
            self.audio_player = pyaudio.PyAudio()
        except Exception as e:
            print(f"Error initializing audio player: {e}")
            self.audio_player = None
            
        self.skipped_pirate_audio = True  # this ensures the last audio from the pirate isn't picked up
        
        try:
            self.pirate_agent = LLMSpeechResponder(eleven_labs_voice_id=eleven_labs_voice_id)
        except Exception as e:
            print(f"Error initializing speech responder: {e}")
            raise
            
        self.image_analysis = ImageAnalysis(prompt=image_prompt)
        self.image_analysis_thread = None
        self.queue = multiprocessing.Queue()
        self.user_description = ""
        
        # Pause/resume controls
        self.is_paused = False
        self.keyboard_thread = None
        self.running = True

    @staticmethod
    def on_open(session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)

    def on_data(self, transcript: aai.RealtimeTranscript):
        if self.is_paused:
            return
            
        if not self.image_analysis_thread:
            self.image_analysis_thread = multiprocessing.Process(
                target=self.image_analysis.take_and_analyse_image,
                args=("", self.queue),
                daemon=True
            )
            self.image_analysis_thread.start()
            print("Thread started")
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            # Create a thread
            if not self.pirate_agent_thread or not self.pirate_agent_thread.is_alive():
                if self.image_analysis_thread and not self.image_analysis_thread.is_alive() and not self.user_description:
                    try:
                        self.user_description = self.queue.get(timeout=5)
                    except:
                        print("Warning: Image analysis timeout, proceeding without user description")
                        self.user_description = ""
                self.pirate_agent_thread = threading.Thread(
                    target=self.pirate_agent.generate,
                    args=(self.user_description, transcript.text),  # Function arguments
                    daemon=True  # Optional: makes thread exit when main program exits
                )
                self.pirate_agent_thread.start()
            else:
                print("Pirate is still speaking")
            print(f"User said: {transcript.text}")
        else:
            # For partial transcripts
            print(transcript.text, end="\r")

    @staticmethod
    def on_error(error: aai.RealtimeError):
        print("An error occurred:", error)

    @staticmethod
    def on_close():
        print("Closing Session")

    def monitor_keyboard(self):
        """Monitor keyboard input for pause/resume commands"""
        print("🏴‍☠️ Type 'p' + Enter to pause/resume, 'r' + Enter to restart, 'q' + Enter to quit")
        
        while self.running:
            try:
                command = input().strip().lower()
                if command == 'p':
                    self.toggle_pause()
                elif command == 'r':
                    self.restart_dialog()
                elif command == 'q':
                    print("\nQuitting...")
                    self.running = False
                    break
                elif command == 'help':
                    print("Commands: 'p' = pause/resume, 'q' = quit")
            except (EOFError, KeyboardInterrupt):
                print("\nQuitting...")
                self.running = False
                break
            except Exception as e:
                print(f"Input error: {e}")
                continue

    def toggle_pause(self):
        """Toggle pause/resume state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            print("\n🏴‍☠️ Pirate PAUSED - Press 'p' to resume, 'q' to quit")
            self.microphone_stream.mute()
        else:
            print("\n🏴‍☠️ Pirate RESUMED - Press 'p' to pause, 'q' to quit")
            self.microphone_stream.unmute()

    def restart_dialog(self):
        print("Restarting Pirate Dialog")
        self.user_description = ""
        self.image_analysis_thread = None
        self.queue = multiprocessing.Queue()
        self.user_transcript = []

    def transcribe(self):
        """
        Start transcribing audio from the microphone for a specified duration.
        """
        try:
            print("🏴‍☠️ Starting Pirate Agent... Press 'p' to pause/resume, 'q' to quit")
            
            # Start keyboard monitoring thread
            self.keyboard_thread = threading.Thread(target=self.monitor_keyboard, daemon=True)
            self.keyboard_thread.start()
            
            self.transcriber.connect()

            # Start streaming audio from the microphone
            self.transcriber.stream(self.microphone_stream)

            return ' '.join(self.user_transcript)
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            self.running = False
        except Exception as e:
            print(f"Error in transcribe: {e}")
            self.running = False

    def cleanup(self):
        """
        Clean up resources when done.
        """
        print("Cleaning up resources...")
        self.running = False
        
        # Stop transcriber
        try:
            if hasattr(self, 'transcriber'):
                self.transcriber.close()
        except Exception as e:
            print(f"Error closing transcriber: {e}")
        
        # Close microphone stream
        try:
            if hasattr(self, 'microphone_stream'):
                self.microphone_stream.close()
        except Exception as e:
            print(f"Error closing microphone: {e}")
            
        # Terminate audio player
        try:
            if hasattr(self, 'audio_player') and self.audio_player:
                self.audio_player.terminate()
        except Exception as e:
            print(f"Error terminating audio player: {e}")
            
        # Join threads
        try:
            if self.pirate_agent_thread and self.pirate_agent_thread.is_alive():
                self.pirate_agent_thread.join(timeout=2)
        except Exception as e:
            print(f"Error joining pirate thread: {e}")
            
        try:
            if self.image_analysis_thread and self.image_analysis_thread.is_alive():
                self.image_analysis_thread.terminate()
        except Exception as e:
            print(f"Error terminating image analysis: {e}")


# Example usage
def run_pirate_agent():
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
                
        print(f"Final transcript: {result}")
    except Exception as e:
        print(f"Error running pirate agent: {e}")
    finally:
        agent.cleanup()


if __name__ == "__main__":
    run_pirate_agent()

