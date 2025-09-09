📧 Spam vs Ham Detection
🔍 Project Overview

This project is a Spam vs Ham Detection System that classifies text messages or emails as either Spam (unwanted messages) or Ham (legitimate messages) using machine learning techniques.

📂 Dataset

Dataset contains labeled SMS/email messages.

Two categories:

Spam → Unwanted/promotional/fraud messages

Ham → Useful/legitimate messages

⚙️ Technologies Used

Python

Libraries:

pandas → Data handling

numpy → Numerical operations

matplotlib / seaborn → Visualization

nltk / scikit-learn → Text preprocessing & ML models

streamlit → Web app deployment

🛠️ Project Workflow
1. Data Preprocessing
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1','v2']]
data.columns = ['label','message']

# Convert labels to numeric
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

2. Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

3. Model Training
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

model = MultinomialNB()
model.fit(X_train_cv, y_train)

y_pred = model.predict(X_test_cv)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

4. Prediction Example
msg = ["Congratulations! You won a lottery. Call now!"]
msg_cv = cv.transform(msg)
print("Prediction:", model.predict(msg_cv))  # 1 = Spam, 0 = Ham

💻 Streamlit App Example (app.py)
import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam vs Ham Detection")
st.write("Enter a message to check if it is **Spam** or **Ham**.")

# User input
user_msg = st.text_area("Message:")

if st.button("Predict"):
    if user_msg.strip() != "":
        msg_cv = cv.transform([user_msg])
        prediction = model.predict(msg_cv)[0]
        if prediction == 1:
            st.error("🚨 This message is **Spam**!")
        else:
            st.success("✅ This message is **Ham** (Safe).")
    else:
        st.warning("Please enter a message first.")

📊 Results

Accuracy: ~95%

Best performance with Naive Bayes

🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/spam_ham_detection.git
cd spam_ham_detection


Install dependencies:

pip install -r requirements.txt


Train and save model (inside spam_detection.py):

import pickle
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))


Run Streamlit app:

streamlit run app.py

📌 Real-Life Applications

Email filtering

SMS spam blocking

Fraud prevention

Customer support automation

✅ Future Improvements

Use advanced models (BERT, LSTM, Transformers)

Multi-language spam detection

Cloud deployment (AWS/GCP/Heroku)
