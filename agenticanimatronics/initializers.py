import os
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

# Load variables from .env file
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=api_key)
agent_id = os.getenv("AGENT_ID")