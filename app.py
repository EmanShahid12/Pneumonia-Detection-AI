import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

# 1. Function to download model from Google Drive
def download_file_from_google_drive(id, destination):
    URL = https://drive.google.com/file/d/128vMTkE6K2mNvNrat-xdjO9YIwTVgxe8/view?usp=drive_link
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: f. f.write(chunk)

# 2. Page Config
st.title("🏥 AI Pneumonia Detector")

# YOUR FILE ID GOES HERE
MODEL_ID = 'YOUR_FILE_ID_HERE' 
MODEL_PATH = 'pneumonia_model.h5'

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model from Google Drive... please wait.'):
        download_file_from_google_drive(MODEL_ID, MODEL_PATH)

# 3. Load Model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 4. Image Upload & Prediction (Same as before)
uploaded_file = st.file_uploader("Upload X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    
    img = image.resize((150, 150))
    img_array = np.array(img.convert('RGB')) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
    st.subheader(f"Result: {result}")
