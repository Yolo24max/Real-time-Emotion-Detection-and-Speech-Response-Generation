import cv2
import numpy as np
from deepface import DeepFace
import time
import threading
import collections
import openai
import pyttsx3
import os
import pygame
import json

# Initialize pygame mixer for audio playback
#pygame.mixer.init()

# Set OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Camera settings
cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("The camera can't be turned on, please check the device connection or permissions.")
        exit(1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
except Exception as e:
    print(f"There was an error when switching on the camera.: {e}")
    exit(1)

# Sliding window to store the most recent emotions
emotion_window = collections.deque(maxlen=2)

# Control the frequency of emotion analysis
last_analysis_time = 0
analysis_interval = 2  # Analyze emotion every 2 seconds
last_emotion = None
is_playing = False  # Check if speech is playing
frame_lock = threading.Lock()  # Thread lock to prevent resource contention
latest_frame = None  # Shared variable to store the latest frame

# Set cooldown time for speech output to avoid rapid switching
cooldown_time = 5  # seconds
last_speech_time = 0

# User data storage (simulating long-term and short-term memory)
USER_DATA_FILE = 'user_data.json'

# Load user data from file
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Save user data to file
def save_user_data(data):
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Get personalized dialogue style based on emotion and user data
def get_persona_based_prompt(emotion, user_data):
    persona = user_data.get('persona', 'friendly')
    return f"As a {persona} AI companion, based on the user's current emotion '{emotion}', generate a personalized response. The response should be brief, warm, and emotional.Limit it to 1-2 sentences.Your response is to a 23-33 year old woman living alone in North America, please think of yourself as a cute intelligent AI pet companion."

def emotion_analysis():
    """Perform emotion analysis and control speech playback (runs in a separate thread)"""
    global last_emotion, last_analysis_time, is_playing, latest_frame, last_speech_time

    user_data = load_user_data()

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

                if most_common_emotion != last_emotion and not is_playing and (current_time - last_speech_time > cooldown_time):
                    last_emotion = most_common_emotion
                    dialog_text = generate_dialog(most_common_emotion, user_data)
                    threading.Thread(target=convert_and_play_speech, args=(dialog_text, most_common_emotion), daemon=True).start()
                    last_speech_time = current_time

                last_analysis_time = current_time

        except Exception as e:
            print(f"Error occurred: {e}")
            log_error(e)

def generate_dialog(emotion, user_data):
    """Generate a short dialog text based on detected emotion"""
    try:
        prompt = get_persona_based_prompt(emotion, user_data)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=[{"role": "system", "content": "You are a friendly assistant."},
                      {"role": "user", "content": prompt}]
        )
        dialog_text = response['choices'][0]['message']['content'].strip()
        print(f"Generated dialog: {dialog_text}")
        return dialog_text
    except Exception as e:
        print(f"Error generating dialog: {e}")
        log_error(e)
        return "Hey, how have you been?"

def convert_and_play_speech(text, emotion):
    """Convert text to speech and play it, adjusting speed and volume based on emotion"""
    global is_playing
    is_playing = True
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  
        engine.setProperty('volume', 1)  

        if emotion == "happy":
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1)
        elif emotion == "sad":
            engine.setProperty('rate', 120)
            engine.setProperty('volume', 0.8)
        elif emotion == "angry":
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1)
        elif emotion == "neutral":
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
        elif emotion == "surprise":
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 1)
        elif emotion == "fear":
            engine.setProperty('rate', 140)
            engine.setProperty('volume', 0.9)

        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"Error with speech synthesis: {e}")
        log_error(e)
    finally:
        is_playing = False

def log_error(error):
    """Log errors to a file for debugging purposes"""
    with open("error_log.txt", "a", encoding='utf-8') as f:
        f.write(f"{time.ctime()}: {error}\n")

def collect_user_feedback(dialog_text):
    """Simulate collecting user feedback"""
    print(f"Please rate the following response:\n'{dialog_text}'")
    feedback = input("Feedback score (0-5): ")
    try:
        feedback = int(feedback)
        if 0 <= feedback <= 5:
            user_data = load_user_data()
            user_data['feedback'] = feedback  
            save_user_data(user_data)
            print(f"Feedback saved: {feedback}")
        else:
            print("Invalid score.")
    except ValueError:
        print("Invalid input.")

# Start emotion analysis thread
analysis_thread = threading.Thread(target=emotion_analysis, daemon=True)
analysis_thread.start()

def capture_frame():
    """Frame capture in a separate thread"""
    global latest_frame
    while True:
        if cap is None or not cap.isOpened():
            print("The camera is not properly turned on to capture frames.")
            time.sleep(1)
            continue
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture frame from camera")
            continue

        with frame_lock:
            latest_frame = frame.copy()

        if last_emotion:
            cv2.putText(frame, f"Emotion: {last_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start frame capture thread
capture_thread = threading.Thread(target=capture_frame, daemon=True)
capture_thread.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Program interrupted by the user")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
