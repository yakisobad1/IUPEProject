import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
import string

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Example dataset
data = {
    "text": [
        "Your account has been suspended. Click here to verify.",
        "Meeting moved to 2pm, see the calendar update.",
        "We need your login credentials to verify your bank account.",
        "Lunch at 1pm?",
        "Unusual login attempt detected. Please reset your password now.",
        "Don't miss the team meeting today!",
    ],
    "label": ["phishing", "legit", "phishing", "legit", "phishing", "legit"]
}

df = pd.DataFrame(data)

#clean text function
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

df["cleaned"] = df["text"].apply(clean_text)

#vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Try on a new email
def predict_email(email_text):
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

# Test it
new_email = "We noticed suspicious activity on your PayPal account. Click here to secure it."
print("Prediction:", predict_email(new_email))
