"""Naive bayes model for tarining on textual data and understanding the intent"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Training Data
training_phrases = [
    # Greetings
    "hello", "hi", "hey there", "good morning", "good evening", "howdy", 

    # Time
    "what's the time", "tell me the time", "give me the current time", "what time is it", 

    # Weather
    "how's the weather", "weather update", "what's the weather like", "is it raining today", 

    # Play Music
    "play some music", "start the music", "play my favorite song", "stop the music", 
    "pause the music", "next song", "previous song", "shuffle the playlist", 
    # Open Applications
    "open calculator", "launch notepad", "start the browser", "open word processor", 
    "run the game", "open the photo viewer", 

    # File Management
    "create a new folder", "delete this file", "move this file to downloads", 
    "copy this document", "rename this file", "open my documents",
    "open a new document", "save this file", "print this page", "upload a file", 
    "download the report", "share this document", "close the current window",  

    # System Commands
    "shut down the computer", "restart my PC", "log me out", "put the computer to sleep", 
    "check for updates", "install the latest updates", "run a virus scan", 
    "clear the cache", "free up disk space", "manage my storage", 
    # Internet Browsing
    "search for news", "open Google", "find recipes online", "show me cat videos", 
    "bookmark this page", "open a new tab", "close the current tab", 
    "go back to the previous page", "refresh the page", "download this image", 
    # Settings
    "open settings", "change the wallpaper", "adjust the volume", "connect to Wi-Fi", 
    "adjust screen brightness", "change the display settings", "set the default printer", 
    "configure keyboard shortcuts", "manage user accounts", 
    # Miscellaneous
    "tell me a joke", "what's on my calendar", "set a reminder for tomorrow", 
    "send an email", "check my notifications", "take a screenshot", 
   "what's the latest news", "tell me a story", "play a game", "check my emails", 
    "schedule a meeting", "set a timer for 10 minutes", "create a to-do list", 
    "search for a tutorial", "find a recipe", "translate this text", 
]

labels = [
    # Corresponding labels
    "greet", "greet", "greet", "greet", "greet", "greet", 

    "time", "time", "time", "time", 

    "weather", "weather", "weather", "weather", 

    "play_music", "play_music", "play_music", "play_music", 
    "play_music", "play_music", "play_music", "play_music",

    "open_app", "open_app", "open_app", "open_app", 
    "open_app", "open_app", 

    "file_management", "file_management", "file_management", 
    "file_management", "file_management", "file_management",
     "file_management", "file_management", "file_management", "file_management", 
    "file_management", "file_management", "file_management", 

    "system_command", "system_command", "system_command", "system_command",
    "system_command", "system_command", "system_command", 
    "system_command", "system_command", "system_command",  

    "internet_browsing", "internet_browsing", "internet_browsing", "internet_browsing", 
    "internet_browsing", "internet_browsing", "internet_browsing", 
    "internet_browsing", "internet_browsing", "internet_browsing", 

    "settings", "settings", "settings", "settings", 
    "settings", "settings", "settings", "settings", 

    "miscellaneous", "miscellaneous", "miscellaneous", 
    "miscellaneous", "miscellaneous", "miscellaneous", 
    "miscellaneous", "miscellaneous", "miscellaneous", "miscellaneous", 
    "miscellaneous", "miscellaneous", "miscellaneous", "miscellaneous",
]

# Training Data
# training_phrases = [
#     "hello", "hi", "hey there",                  # Greet
#     "what's the time", "tell me the time",       # Time
#     "how's the weather", "weather update",       # Weather
#     "play some music", "start the music"         # Play Music
# ]
# labels = ["greet", "greet", "greet", "time", 
#         "time", "weather", "weather", "play_music", "play_music"]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_phrases)

model = MultinomialNB()
model.fit(X_train,labels)

with open('naiveModel.pkl','wb') as f:
    pickle.dump(model,f)
with open("vectorizer.pkl",'wb') as f:
    pickle.dump(vectorizer,f)    

