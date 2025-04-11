# AgenticAnimatronics

## ABOUT
This package contains different Characters for use in animatronics. 
These characters are built on speech to text, text to speech, and an LLM which writes the dialog.
The purpose of this package is to provide ready to use characters for my dad to implement in his 
halloween party.

## QUICKSTART
Here is a quick start guide to running your agent
As a prerequisite create an elevenlabs account: https://elevenlabs.io
### installations
1. install homebrew https://brew.sh
2. install python3.12+ https://www.python.org/downloads/
3. install poetry https://python-poetry.org/docs/#installing-with-the-official-installer
4. open terminal and type: brew install portaudio19
### Downloading this repository
1. In terminal: cd ~/Documents
2. In terminal: git clone https://github.com/samquiring/AgenticAnimatronics.git
3. In terminal: cd ~/Documents/AgenticAnimatronics
3. In terminal: poetry install
4. In terminal: cp .env_example .env
### Updating the .env file
1. Open .env (it should be in Documents/AgenticAnimatronics/.env)
2. go to https://elevenlabs.io/app/settings/api-keys and create an api key. Copy the key
3. Paste the key into the .env file under ELEVENLABS_API_KEY (so ELEVENLABS_API_KEY=(your key))
### Running the pirate agent
1. In terminal: cd ~/Documents/AgenticAnimatronics
2. In terminal: poetry run pirate-agent
3. To exit out of the conversation In terminal: control+c (so hit the control button and c at the same time)

