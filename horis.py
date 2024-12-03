import os
import speech_recognition as sr
import pyttsx3
import pvporcupine
import pyaudio
import struct
import pickle
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

porcupine_api_key = os.getenv('PORCUPINE_API_KEY')

#initailize the listener
recognizer = sr.Recognizer()

#initialize the text to speech engine
engine = pyttsx3.init()
with open('naiveModel.pkl','rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)    


def classify_intent(command):
    """Classify intent of command based on Model"""
    vectorized_input = vectorizer.transform([command])
    print(vectorized_input)
    return model.predict(vectorized_input)[0]

def greet(user_input):
    """Greet the user based on input"""
    if user_input == "hello":
        return "Hello, how are you today?"
    if user_input == "hi":
        return "Hi, how are you today?"
    if user_input == "hey":
        return "Hey, how are you today?"
    return "Hello, how are you today?"

def tell_time(user_input):
    """Tell the user the current time"""
    ## create a model to identify city ?
    return datetime.now().strftime("%H:%M")

def tell_weather():
    """Tell the user the current weather"""
    return "returning from function weather"
def play_music():
    """Play music"""
    return "returning from function music"
def play_video():
    return "returning from function video"
def open_app():
    return "returning from function app"
def file_management():
    return "returning from function file"
def system_command():
    return "returning from function system"
def internet_browsing():
    return internet_browsing
def settings():
    return "returning from function settings"
def miscellaneous():
    return "returning from function miscellaneous"


# Intents = [ greet, time, weather, play_music, open_app, file_management, 
# system command, internet_browsing, settings, miscellaneous]

intent_map = {
    "greet": greet,
    "play_music":play_music,
    "weather": tell_weather,
    "time":tell_time,
    "open_app":open_app,
    "file_management":file_management,
    "system_command":system_command,
    "internet_browsing":internet_browsing,
    "settings":settings,
    "miscellaneous":miscellaneous
}


def listen():
    """
    Listen For user voice
    """
    try:
        print('listenting...')
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
    except sr.WaitTimeoutError:
        print("Timeout: No speech detected." )   
    except sr.RequestError as e:
        print(f"Error with the recognition service: {e}")

def speak(text):
    """
    Speak text 
    """
    engine.say(text)
    engine.runAndWait()

def assistant():
    """Assistant Function """
    # speak('Hello , i am horis, How may i assist you today?')

    while True:
        user_input = listen()
        if "exit" in user_input:
            break
        print(f"You said: {user_input}")
        intent = classify_intent(user_input.lower())
        print("intent---->",intent)
        action = intent_map.get(intent,lambda:"unable to understand")
        print(action(user_input))
        speak(action(user_input))
        


def porcupine_wake_word():
    porcupine = pvporcupine.create(
        access_key= porcupine_api_key,  # Replace with your Picovoice API key
        keyword_paths=["hey-horis_en_windows_v3_0_0.ppn"]
    )

    #initialize pyaudio for audio output
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    print("Listening for wake word...")

    try:
        while True:
            pcm = audio_stream.read(porcupine.frame_length,exception_on_overflow = False)
            pcm = struct.unpack_from ("h"*porcupine.frame_length,pcm)

            if porcupine.process(pcm) >= 0: 
                print("wake word Detected")
                assistant()
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()                

# Run the assistant
porcupine_wake_word()


# -------------------------------------------old iterations-------------------------------------------------

# def assistant():
#     """Assistant Function """
#     # speak('Hello , i am horis, How may i assist you today?')

#     while True:
#         user_input = listen()
#         print(f"You said: {user_input}")
#         if user_input is None:
#             speak("my name is Horis. What is your name?")

#         if "name" in user_input.lower():
#             speak("my name is Horis. What is your name?")
        
#         elif "exit" in user_input.lower():
#             speak("goodbye, see you soon")
#             break

#         # if "hello" in user_input.lower():
#         #     speak("hi , my name is horis, how may I assist you today?")   
#         #
#         elif "pilli" in  user_input.lower() or 'vaibhav' in  user_input.lower() :
#             speak("Should i say Bow bow?")
#         elif "time" in user_input.lower():
#             current_time = datetime.now().strftime("%H:%M")
#             speak(f"The time is {current_time}.")

#         else:
#             speak("I'm sorry, I don't understand that command.")  

