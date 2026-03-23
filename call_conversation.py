import pyttsx3
from pydub import AudioSegment
import os
import time


def speak(text, voice_index=0, rate=170):

    # 🔹 Create NEW engine each time (important)
    engine = pyttsx3.init()

    voices = engine.getProperty('voices')

    if len(voices) > voice_index:
        engine.setProperty('voice', voices[voice_index].id)

    engine.setProperty('rate', rate)

    temp_file = f"temp_{time.time()}.wav"

    engine.save_to_file(text, temp_file)
    engine.runAndWait()
    engine.stop()

    audio = AudioSegment.from_wav(temp_file)
    os.remove(temp_file)

    return audio


conversation = AudioSegment.empty()

# 😊 Happy
conversation += speak(
    "Hey! I'm so happy you called. I was just thinking about you!",
    0,
    185
) + AudioSegment.silent(duration=600)

# 😄 Cheerful
conversation += speak(
    "Really? That makes me smile. I had a great day today.",
    1,
    180
) + AudioSegment.silent(duration=600)

# 😟 Concerned
conversation += speak(
    "You sound tired though… Is everything okay?",
    0,
    155
) + AudioSegment.silent(duration=600)

# 😢 Sad
conversation += speak(
    "Not really… I got some bad news from work.",
    1,
    140
) + AudioSegment.silent(duration=600)

# 🤝 Supportive
conversation += speak(
    "Oh no… I’m really sorry. Do you want to talk about it?",
    0,
    150
) + AudioSegment.silent(duration=600)

# 😌 Relieved
conversation += speak(
    "Thanks… It means a lot that you’re listening.",
    1,
    165
)

conversation.export("call_conversation.wav", format="wav")

print("✅ Created: call_conversation.wav")