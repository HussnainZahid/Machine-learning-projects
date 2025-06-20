# gui.py
import tkinter as tk
from tkinter import messagebox
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load model & vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict_spam():
    msg = text_entry.get("1.0", tk.END)
    if not msg.strip():
        messagebox.showwarning("Input Error", "Please enter a message.")
        return
    cleaned = clean_text(msg)
    vector = vectorizer.transform([cleaned]).toarray()
    result = model.predict(vector)[0]
    if result == 1:
        result_label.config(text="🚫 Spam", fg="red")
    else:
        result_label.config(text="✅ Not Spam", fg="green")

# UI setup
root = tk.Tk()
root.title("Spam Email Detector")
root.geometry("400x300")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Enter Message:", font=("Arial", 12)).pack(pady=10)
text_entry = tk.Text(root, height=5, width=40)
text_entry.pack()

tk.Button(root, text="Detect", command=predict_spam, bg="#4CAF50", fg="white", width=20).pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=10)

root.mainloop()
