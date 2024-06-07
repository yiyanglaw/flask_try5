from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
import string
import numpy as np

app = Flask(__name__)

# Load the data for training
data = pd.read_csv('sms_3.csv')

# Fill missing values with an empty string
data['Message'] = data['Message'].fillna('')

data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

# Create a pipeline with CountVectorizer and MultinomialNB
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Training the model
clf.fit(X_train, y_train)

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    text = request.form['text']
    processed_text = preprocess_text(text)
    prediction = clf.predict([processed_text])[0]
    if prediction == 0:
        result = 'Ham (Not Spam)'
    else:
        result = 'Spam'
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
