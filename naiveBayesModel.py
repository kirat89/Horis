"""Naive bayes model for tarining on textual data and understanding the intent"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


# Training Data
training_phrases = [
    "hello", "hi", "hey there",                  # Greet
    "what's the time", "tell me the time",       # Time
    "how's the weather", "weather update",       # Weather
    "play some music", "start the music"         # Play Music
]
labels = ["greet", "greet", "greet", "time", 
        "time", "weather", "weather", "play_music", "play_music"]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_phrases)

model = MultinomialNB()
model.fit(X_train,labels)

with open('naiveModel.pkl','wb') as f:
    pickle.dump(model,f)
with open("vectorizer.pkl",'wb') as f:
    pickle.dump(vectorizer,f)    

