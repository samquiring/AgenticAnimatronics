
import pyaudio
from elevenlabs import ElevenLabs, stream, VoiceSettings

from agenticanimatronics.initializers import eleven_labs_key
from agenticanimatronics.llm import LLMHandler
from agenticanimatronics.pirate_chatbot_module import PirateChatBot


class LLMSpeechResponder:
    def __init__(self, eleven_labs_voice_id):
        """
        Takes user input and generates an llm response and then uses elevenlabs to stream the response as speech.
        :param eleven_labs_voice_id:
        """
        try:
            self.llm = LLMHandler()
            self.pirate_chatbot = PirateChatBot()
            self.eleven_labs_voice_id = eleven_labs_voice_id
            self.eleven_labs_client = ElevenLabs(api_key=eleven_labs_key)
            self.conversation_history = []
            self.interrupt_lock = False  # A lock to ignore user conversation while agent is speaking

            # Initialize audio playback components
            self.audio_player = pyaudio.PyAudio()
        except Exception as e:
            print(f"Error initializing LLMSpeechResponder: {e}")
            raise

    def text_to_speech_stream(self, text: str) -> bytes:
        """
        Convert text to speech using ElevenLabs API and return a stream of audio bytes.
        """
        print("Converting to speech")
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Perform the text-to-speech conversion
                response = self.eleven_labs_client.generate(
                    voice=self.eleven_labs_voice_id,
                    optimize_streaming_latency=3,
                    output_format="mp3_22050_32",
                    text=text,
                    stream=True,
                    voice_settings=VoiceSettings(
                        stability=0.5,  # A balanced setting for natural but consistent voice
                        similarity_boost=0.75,
                        style=0.0,
                        use_speaker_boost=True,
                        speed=1.0,
                    ),
                )
                return stream(response)
            except Exception as e:
                print(f"ElevenLabs API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("Could not generate speech - pirate voice is silent")
                    return None
                import time
                time.sleep(1)

    def update_conversational_history(self, user_response: str, assistant_response: str):
        """
        Update the conversation history with the user's input and the assistant's response.
        """
        print("Updating conversational history")
        self.conversation_history.append({"role": "user", "content": user_response})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

    def generate(self, user_description: str, user_response: str):
        print(f"Generating response for user input: {user_response}")

        try:
            # Get response from pirate chatbot
            pirate_response = self.pirate_chatbot.forward(
                history=self.conversation_history,
                user_prompt=user_response,
                user_description=user_description,
            )

            print(f"Pirate responds: {pirate_response}")

            # Convert the response to speech and get the audio stream
            # microphone_stream.mute()
            audio_stream = self.text_to_speech_stream(pirate_response)
            if audio_stream is None:
                print("Speech generation failed - continuing without audio")
            # microphone_stream.unmute()
            
            # Update conversation history
            self.update_conversational_history(user_response, pirate_response)
        except Exception as e:
            print(f"Error in generate method: {e}")
            fallback_response = "Arr, something went wrong with me voice, matey!"
            self.update_conversational_history(user_response, fallback_response)