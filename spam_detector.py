import streamlit as st
import pickle

# Load model and vectorizer
MODEL_PATH = 'spam.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

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

def main():
    st.title("Email Spam Classification Application")
    st.write("A Machine Learning application to classify emails as spam or not spam.")

    model, cv = load_model_and_vectorizer()
    if not model or not cv:
        return

    st.subheader("Classification")
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

if __name__ == "__main__":
    main()
