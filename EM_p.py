import cv2
import numpy as np
from deepface import DeepFace
import time
import threading
import collections
import openai
import pyttsx3  # Text-to-speech library
import os
import pygame

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Set OpenAI API key (make sure the key is valid)
openai.api_key = "YOUR_OPENAI_KEY"

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Sliding window to store the last 2 emotions
emotion_window = collections.deque(maxlen=2)

# Control emotion analysis rate
last_analysis_time = 0
analysis_interval = 2  # Analyze emotion every 2 seconds
last_emotion = None
is_playing = False  # To check if speech is playing
frame_lock = threading.Lock()  # Thread lock to prevent resource contention
latest_frame = None  # Shared variable to store the latest frame

# Define cooldown for speech output to avoid too frequent changes
cooldown_time = 5  # seconds
last_speech_time = 0

def emotion_analysis():
    """Perform emotion analysis and control speech playback (running in a separate thread)"""
    global last_emotion, last_analysis_time, is_playing, latest_frame, last_speech_time

    while True:
        time.sleep(0.1)

        with frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        current_time = time.time()
        if current_time - last_analysis_time < analysis_interval:
            continue

        try:
            small_frame = cv2.resize(frame_copy, (320, 240))
            results = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')

            if results:
                detected_emotion = results[0]['dominant_emotion']
                emotion_window.append(detected_emotion)
                most_common_emotion = collections.Counter(emotion_window).most_common(1)[0][0]

                print(f"Detected emotion: {detected_emotion}, Smoothed emotion: {most_common_emotion}")

                # Check if emotion is changed and if cooldown period has passed
                if most_common_emotion != last_emotion and not is_playing and (current_time - last_speech_time > cooldown_time):
                    last_emotion = most_common_emotion
                    dialog_text = generate_dialog(most_common_emotion)
                    threading.Thread(target=convert_and_play_speech, args=(dialog_text, most_common_emotion), daemon=True).start()
                    last_speech_time = current_time

                last_analysis_time = current_time

        except Exception as e:
            print(f"Error occurred: {e}")
            log_error(e)

def generate_dialog(emotion):
    """Generate short dialog text based on the detected emotion (limit to 1-2 sentences)"""
    try:
        prompt = f"Create a natural and conversational response based on the user's emotion: '{emotion}'. Keep the response casual, short, and engaging. Limit it to 1-2 sentences.Your response is to a 23-33 year old woman living alone in North America, please think of yourself as a cute intelligent AI pet companion."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use an appropriate model
            messages=[{"role": "system", "content": "You are a friendly assistant."},
                      {"role": "user", "content": prompt}]
        )
        dialog_text = response['choices'][0]['message']['content'].strip()
        print(f"Generated dialog: {dialog_text}")
        return dialog_text
    except Exception as e:
        print(f"Error generating dialog: {e}")
        log_error(e)
        return "Hey there! How are you doing?"

def convert_and_play_speech(text, emotion):
    """Convert text to speech and play it using pyttsx3, adjusting voice speed/pitch based on emotion"""
    global is_playing
    is_playing = True
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Set speech rate
        engine.setProperty('volume', 1)  # Set volume

        # Adjust voice properties based on emotion
        if emotion == "happy":
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1)
        elif emotion == "sad":
            engine.setProperty('rate', 120)
            engine.setProperty('volume', 0.8)

        # Convert text to speech and play it
        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"Error with speech synthesis: {e}")
        log_error(e)
    finally:
        is_playing = False

def log_error(error):
    """Log errors to a file for debugging"""
    with open("error_log.txt", "a", encoding='utf-8') as f:
        f.write(f"{time.ctime()}: {error}\n")

# Start the emotion analysis thread
analysis_thread = threading.Thread(target=emotion_analysis, daemon=True)
analysis_thread.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera")
            continue

        with frame_lock:
            latest_frame = frame.copy()

        if last_emotion:
            cv2.putText(frame, f"Emotion: {last_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Analysis', frame)

       
except KeyboardInterrupt:
    print("User interrupted the program")
finally:
    cap.release()
    cv2.destroyAllWindows()
