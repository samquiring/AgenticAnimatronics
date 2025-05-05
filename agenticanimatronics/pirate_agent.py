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
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            end_utterance_silence_threshold=700,
            disable_partial_transcripts=True
        )
        self.microphone_stream = MutableMicrophoneStream(sample_rate=16000)
        self.user_transcript = []
        self.pirate_agent_thread = None
        # Initialize audio playback components
        self.audio_player = pyaudio.PyAudio()
        self.skipped_pirate_audio = True  # this ensures the last audio from the pirate isn't picked up
        self.pirate_agent = LLMSpeechResponder(eleven_labs_voice_id=eleven_labs_voice_id)
        self.image_analysis = ImageAnalysis(prompt=image_prompt)
        self.image_analysis_thread = None
        self.queue = multiprocessing.Queue()
        self.user_description = ""

    @staticmethod
    def on_open(session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not self.image_analysis_thread:
            self.image_analysis_thread = multiprocessing.Process(
                target=self.image_analysis.take_and_analyse_image,
                args=("",self.queue),
                daemon=True
            )
            self.image_analysis_thread.start()
            print("Thread started")
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            # Create a thread
            if not self.pirate_agent_thread or not self.pirate_agent_thread.is_alive():
                if self.image_analysis_thread and not self.image_analysis_thread.is_alive():
                    self.user_description = self.queue.get()
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

    def transcribe(self):
        """
        Start transcribing audio from the microphone for a specified duration.
        """
        self.transcriber.connect()

        # Start streaming audio from the microphone
        self.transcriber.stream(self.microphone_stream)

        return ' '.join(self.user_transcript)

    def cleanup(self):
        """
        Clean up resources when done.
        """
        if hasattr(self, 'audio_player') and self.audio_player:
            self.audio_player.terminate()


# Example usage
def run_pirate_agent():
    agent = PirateAgent()
    try:
        print("Listening...")
        result = agent.transcribe()
        print(f"Final transcript: {result}")
    finally:
        agent.cleanup()


if __name__ == "__main__":
    run_pirate_agent()

