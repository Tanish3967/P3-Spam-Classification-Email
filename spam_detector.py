import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import os

# File paths for saved model and vectorizer
MODEL_PATH = 'spam.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# Function to train the model
def train_model():
    st.title("Train Spam Detection Model")
    st.write("This section allows you to train a spam detection model.")

    # Upload the dataset
    uploaded_file = st.file_uploader("Upload your spam dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        try:
            # Try reading the dataset
            data = pd.read_csv(uploaded_file, encoding='latin-1')
            st.write("Dataset Preview:")
            st.dataframe(data.head())
        except UnicodeDecodeError:
            st.error("Error reading file. Please ensure it is encoded in UTF-8 or Latin-1.")
            return

        # Validate dataset
        if 'class' in data.columns and 'message' in data.columns:
            data = data[['class', 'message']]
            data.columns = ['label', 'text']
            data['label'] = data['label'].map({'ham': 0, 'spam': 1})
            
            # Feature extraction
            cv = CountVectorizer()
            X = cv.fit_transform(data['text'])
            y = data['label']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = MultinomialNB()
            model.fit(X_train, y_train)

            # Evaluate model
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.success(f"Model trained successfully with an accuracy of {accuracy:.2f}")

            # Save model and vectorizer
            with open(MODEL_PATH, 'wb') as model_file:
                pickle.dump(model, model_file)
            with open(VECTORIZER_PATH, 'wb') as vectorizer_file:
                pickle.dump(cv, vectorizer_file)
            st.success("Model and vectorizer saved successfully.")
        else:
            st.error("Dataset must contain 'class' and 'message' columns.")
    else:
        st.info("Please upload a dataset to begin training.")


# Function to load model and vectorizer
def load_model_and_vectorizer():
    try:
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
            cv = pickle.load(vectorizer_file)
        return model, cv
    except FileNotFoundError:
        st.error("Model or vectorizer not found. Please train the model first.")
        return None, None

# Function to classify emails
def classify_email():
    st.title("Email Spam Classification")
    st.write("This section allows you to classify emails as spam or not spam.")

    model, cv = load_model_and_vectorizer()
    if not model or not cv:
        return

    user_input = st.text_area("Enter an email to classify", height=150)
    if st.button("Classify"):
        if user_input:
            data = [user_input]
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            if result[0] == 0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is A Spam Email")
        else:
            st.warning("Please enter an email to classify.")

# Main app
def main():
    st.sidebar.title("Spam Detection App")
    app_mode = st.sidebar.radio("Choose Mode", ["Train Model", "Classify Email"])

    if app_mode == "Train Model":
        train_model()
    elif app_mode == "Classify Email":
        classify_email()

if __name__ == "__main__":
    main()
