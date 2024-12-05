import random
import time

# Streamed response emulator
def response_generator(state = "default"):
    response = ""
    if state == "default":
        response = random.choice(
            [
                "Default reply",
                "Default are we?",
                "Default's up?",
            ]
        )
    else:
        response = random.choice(
            [
                f"State detected: {state}",
                f"Where are we? At state {state}",
                f"What's up in state {state}?",
            ]
        )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

