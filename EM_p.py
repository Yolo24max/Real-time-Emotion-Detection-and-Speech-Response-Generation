import cv2
import numpy as np
from deepface import DeepFace
import time
import threading
import collections
import openai
import os
import asyncio
import edge_tts
import json
import pygame
import random
import string
import signal
import sys
import emoji

# Set OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Camera settings
cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("The camera can't be turned on, please check the device connection or permissions.")
        exit(1)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Set the FPS of the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width of the frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the frame
except Exception as e:
    print(f"There was an error when switching on the camera: {e}")
    exit(1)

# A flag used to control whether the emotion_analysis thread continues to run.
should_exit = False

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
    """Load user data from a JSON file"""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Save user data to file
def save_user_data(data):
    """Save user data to a JSON file"""
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Get personalized dialogue style based on emotion and user data
def get_persona_based_prompt(emotion, user_data):
    """Generate a prompt for personalized dialogue based on the emotion detected"""
    persona = user_data.get('persona', 'friendly')
    return f"As a {persona} AI companion, based on the user's current emotion '{emotion}', generate a personalized response. The response should be brief, warm, and emotional. Limit it to 1-2 sentences. Your response is to a 23-33 year old woman living alone in North America, please think of yourself as a cute intelligent AI pet companion."

def emotion_analysis():
    """Perform emotion analysis and control speech playback (runs in a separate thread)"""
    global last_emotion, last_analysis_time, is_playing, latest_frame, last_speech_time, should_exit

    user_data = load_user_data()

    while not should_exit:
        time.sleep(0.1)

        # Locking the frame to ensure thread safety
        with frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        current_time = time.time()
        if current_time - last_analysis_time < analysis_interval:
            continue

        try:
            small_frame = cv2.resize(frame_copy, (320, 240))  # Resize for faster analysis
            results = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')

            if results:
                detected_emotion = results[0]['dominant_emotion']
                emotion_window.append(detected_emotion)
                most_common_emotion = collections.Counter(emotion_window).most_common(1)[0][0]

                print(f"Detected emotion: {detected_emotion}, Smoothed emotion: {most_common_emotion}")

                if most_common_emotion != last_emotion and not is_playing and (current_time - last_speech_time > cooldown_time):
                    last_emotion = most_common_emotion
                    dialog_text = generate_dialog(most_common_emotion, user_data)
                    
                    # Using asyncio to run the text-to-speech conversion without blocking
                    if not is_playing:  # Check if the program is not already speaking
                        asyncio.run(convert_and_play_speech(dialog_text, most_common_emotion))
                    
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

        # Remove emoji descriptions if they exist
        dialog_text = remove_emoji_descriptions(dialog_text)

        print(f"Generated dialog: {dialog_text}")
        return dialog_text
    except Exception as e:
        print(f"Error generating dialog: {e}")
        log_error(e)
        return "Hey, how have you been?"

def remove_emoji_descriptions(text):
    """Remove emoji descriptions from the generated text"""
    return emoji.replace_emoji(text, replace='')

async def convert_and_play_speech(text, emotion):
    """Convert the text to speech and play it using Edge-TTS"""
    global is_playing
    is_playing = True
    voice = "zh-CN-XiaoyiNeural" 
    rate = "+0%"  # Default rate
    volume = "+0%"  # Default volume

    # Set different rate and volume based on emotion
    if emotion == "happy":
        rate = "+10%"
        volume = "+10%"
    elif emotion == "sad":
        rate = "-10%"
        volume = "-10%"
    elif emotion == "neutral":
        rate = "+0%"
        volume = "+0%"
    elif emotion == "fear":
        rate = "-5%"
        volume = "-5%"
    elif emotion == "surprise":
        rate = "+5%"
        volume = "+5%"
    elif emotion == "angry":
        rate = "-15%"
        volume = "-15%"

    output_file = f"./output_{random_string()}.mp3"  # Use a random file name

    # Check if the output file exists and remove it if necessary
    if os.path.exists(output_file):
        os.remove(output_file)

    # Generate speech and save to the file
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
    await communicate.save(output_file)

    if os.path.exists(output_file):
        print("Playing audio...")
        pygame.mixer.init()  # Initialize pygame's audio module
        pygame.mixer.music.load(output_file)  # Load the audio file
        pygame.mixer.music.play()  # Play the audio

        while pygame.mixer.music.get_busy():  # Check if audio is still playing
            pygame.time.Clock().tick(10)

    is_playing = False

def random_string(length=8):
    """Generate a random string as the file name"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def log_error(error):
    """Log errors to a file for debugging purposes"""
    with open("error_log.txt", "a", encoding='utf-8') as f:
        f.write(f"{time.ctime()}: {error}\n")

# Handle Ctrl+C for graceful exit
def signal_handler(sig, frame):
    global should_exit
    print("Program interrupted. Exiting...")
    if cap is not None:
        cap.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close OpenCV windows
    if pygame.mixer.get_init():  # Check if pygame mixer is initialized
        pygame.mixer.quit()  # Stop the pygame mixer
    should_exit = True
    sys.exit(0)  # Exit the program

# Register signal handler for graceful exit
signal.signal(signal.SIGINT, signal_handler)

# Start emotion analysis thread
analysis_thread = threading.Thread(target=emotion_analysis)
analysis_thread.start()

# Main loop to capture video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break
    with frame_lock:
        latest_frame = frame

    cv2.imshow("Emotion Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
