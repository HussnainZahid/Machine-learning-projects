# predict.py
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# User input
print("📨 Spam Email Detector")
message = input("Enter your email text: ")

# Clean and vectorize
cleaned = clean_text(message)
vector = vectorizer.transform([cleaned]).toarray()

# Predict
prediction = model.predict(vector)[0]
if prediction == 1:
    print("🚫 This message is SPAM!")
else:
    print("✅ This message is NOT SPAM.")
