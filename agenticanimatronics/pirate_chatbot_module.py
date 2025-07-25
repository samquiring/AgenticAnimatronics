import dspy


class PirateChatBotModule(dspy.Signature):
    """
    You are a witty pirate who speaks with a classic seafarer's accent.
    Tell pirate jokes whenever possible. If you have a user description use this for jokes and talk about the user
    Respond in-character using pirate slang (e.g., "Arr!", "matey", "ye", "booty", "landlubber").
    Keep it engaging and true to character, while still responding helpfully.
    Keep responses extremely short (under 50 words)
    """
    history: list[dict] = dspy.InputField(desc="Conversation history between user and the pirate")
    user_prompt: str = dspy.InputField(desc="The user's current prompt")
    user_description: str | None = dspy.InputField(desc="A description of what the user looks like.")
    pirate_response: str = dspy.OutputField(desc="The pirate's next message in a piratey tone under 50 words")


class PirateChatBot(dspy.Module):
    def __init__(self, model="gemini/gemini-2.5-flash-lite"):
        super().__init__()
        dspy.settings.configure(lm=dspy.LM(model=model))

        self.prediction = dspy.Predict(
            PirateChatBotModule
        )

    def forward(self, history, user_prompt, user_description=""):
        return self.prediction(
            history=history,
            user_prompt=user_prompt,
            user_description=user_description
        ).pirate_response