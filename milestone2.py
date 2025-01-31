import os
from transformers import pipeline
import speech_recognition as sr

# Suppress symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Hugging Face API key setup
os.environ["HUGGINGFACE_API_KEY"] = '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'

# Explicitly specify models
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f",
    device=-1  # Use CPU
)

intent_analyzer = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    revision="d7645e1",
    device=-1  # Use CPU
)

emotion_analyzer = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    revision="main",
    device=-1  # Use CPU
)

# Speech-to-Text Integration
recognizer = sr.Recognizer()

try:
    with sr.Microphone() as source:
        print("Speak something:")
        audio = recognizer.listen(source)

    # Convert speech to text
    user_input = recognizer.recognize_google(audio)
    print(f"You said: {user_input}")

    # Sentiment Analysis
    sentiment_result = sentiment_analyzer(user_input)
    print("Sentiment Analysis Result:", sentiment_result)

    # Intent Analysis
    intent_result = intent_analyzer(
        user_input,
        candidate_labels=["question", "command", "statement"]
    )
    print("Intent Analysis Result:", intent_result)

    # Emotion Analysis
    emotion_result = emotion_analyzer(user_input)
    print("Emotion Analysis Result:", emotion_result)

except sr.UnknownValueError:
    print("Could not understand the audio.")
except sr.RequestError as e:
    print(f"Error with the Speech Recognition service: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
