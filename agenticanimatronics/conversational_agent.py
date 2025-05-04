import threading
import pyaudio
import assemblyai as aai
import assemblyai.extras

from agenticanimatronics.image_analysis import ImageAnalysis
from agenticanimatronics.initializers import assembly_ai_key
from agenticanimatronics.mutable_microphone_stream import MutableMicrophoneStream
from agenticanimatronics.pirate_agent import PirateAgent

aai.settings.api_key = assembly_ai_key


class ConversationalAgent:
    def __init__(self, eleven_labs_voice_id="Myn1LuZgd2qPMOg9BNtC"):
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
        self.pirate_agent = PirateAgent(eleven_labs_voice_id=eleven_labs_voice_id)
        self.image_analysis = ImageAnalysis()
        self.image_analysis_thread = None

    @staticmethod
    def on_open(session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not self.image_analysis_thread:
            self.image_analysis_thread = threading.Thread(
                target=self.image_analysis.take_and_analyse_image(),
                args=(),
                daemon=True
            )
            self.image_analysis_thread.start()
            print("Thread started")
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            # Create a thread
            # TODO: Convert to proper multithreading for non-blocking operation
            # if not self.pirate_agent_thread or not self.pirate_agent_thread.is_alive():
            #     if self.image_analysis_thread and not self.image_analysis_thread.is_alive():
            #         user_description = self.image_analysis.analysis
            #     else:
            #         user_description = ""
            #     self.pirate_agent_thread = threading.Thread(
            #         target=self.pirate_agent.generate,
            #         args=(user_description, transcript.text),  # Function arguments
            #         daemon=True  # Optional: makes thread exit when main program exits
            #     )
            #     self.pirate_agent_thread.start()
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
def main():
    agent = ConversationalAgent()
    try:
        print("Listening...")
        result = agent.transcribe()
        print(f"Final transcript: {result}")
    finally:
        agent.cleanup()


if __name__ == "__main__":
    main()
#
# conversational_agent = ConversationalAgent()
# conversational_agent.text_to_speech_stream("Testing to see if this works.")
# # asyncio.run(conversational_agent.transcribe(10))

