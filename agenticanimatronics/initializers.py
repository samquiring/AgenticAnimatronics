import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
eleven_labs_key = os.getenv("ELEVENLABS_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
assembly_ai_key = os.getenv("ASSEMBLY_AI_API_KEY")
agent_id = os.getenv("AGENT_ID")
