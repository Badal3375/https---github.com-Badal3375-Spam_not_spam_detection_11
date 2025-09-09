import pandas as pd
import nltk
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Make sure NLTK resources are available
nltk.download('stopwords')
stop_words = stopwords.words('english')
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    words = []
    for word in text.split():
        if word not in stop_words and word not in string.punctuation:
            words.append(ps.stem(word))
    return " ".join(words)

# 📂 Load your dataset (replace with your CSV path)
# Dataset should have two columns: ['label', 'message']
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1','v2']]
df = df.rename(columns={'v1':'label', 'v2':'message'})

# Preprocess text
df['transformed'] = df['message'].apply(transform_text)

# Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transformed'])
y = df['label'].map({'ham':0, 'spam':1})

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save vectorizer & model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and Vectorizer saved successfully!")


