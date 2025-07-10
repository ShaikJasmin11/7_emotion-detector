# app.py

import streamlit as st
import joblib

st.set_page_config(page_title="😄 Emotion Detector", layout="centered")

st.title("😄 Emotion Detection from Text")
st.write("Enter a sentence and get the emotion behind it!")

model = joblib.load("models/emotion_model.pkl")

user_input = st.text_area("Type your sentence here:")

if st.button("Detect Emotion"):
    prediction = model.predict([user_input])[0]
    st.success(f"🧠 Detected Emotion: **{prediction.upper()}**")

if st.button("Reset"):
    st.rerun()
