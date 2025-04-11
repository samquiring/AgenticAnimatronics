import signal

from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

from agenticanimatronics.initializers import client, agent_id, api_key


def run_pirate_agent():
    conversation = Conversation(
        # API client and agent ID.
        client,
        agent_id,
        # Assume auth is required when API_KEY is set.
        requires_auth=bool(api_key),
        # Use the default audio interface.
        audio_interface=DefaultAudioInterface(),
        # Simple callbacks that print the conversation to the console.
        callback_agent_response=lambda response: print(f"Agent: {response}"),
        callback_agent_response_correction=lambda original, corrected: print(f"Agent: {original} -> {corrected}"),
        callback_user_transcript=lambda transcript: print(f"User: {transcript}"),
    )
    conversation.start_session()
    signal.signal(signal.SIGINT, lambda sig, frame: conversation.end_session())
    conversation_id = conversation.wait_for_session_end()
    print(f"Conversation ID: {conversation_id}")



