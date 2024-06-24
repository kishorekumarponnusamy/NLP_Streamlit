import joblib
import streamlit as st

saved_model = joblib.load("news-classification-model.pkl")

classifer_labels = {0: "Tech", 1: "Business",2:"Sports",3:"Entertainment",4:"Politics"}

st.title("Text Classification ")
user_input = st.text_input("Enter a sentence:")

if st.button("Predict"):
    prediction = saved_model.predict([user_input])
    st.info(f"Predicted Classification : {classifer_labels[prediction[0]]}")
    

