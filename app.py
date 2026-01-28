import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Title
st.title("🏥 AI Pneumonia Detector")
st.write("Upload a Chest X-ray image to get an AI prediction.")

# 2. Load the trained model (Make sure 'pneumonia_model.h5' is in your GitHub)
@st.cache_resource
def load_my_model():
    # Only load the model, don't try to download datasets here
    return tf.keras.models.load_model('pneumonia_model.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error("Model file not found! Please make sure 'pneumonia_model.h5' is uploaded to GitHub.")

# 3. Image Upload Utility
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    
    # 4. Preprocessing (Match the training size: 150x150)
    img = image.resize((150, 150))
    img_array = np.array(img.convert('RGB')) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Prediction Logic
    prediction = model.predict(img_array)
    result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
    confidence = prediction[0][0] if result == "PNEUMONIA" else 1 - prediction[0][0]
    
    # 6. Show Results
    st.subheader(f"Detection Result: {result}")
    st.write(f"Confidence Level: {confidence*100:.2f}%")
    
    # Simple advice based on result
    if result == "PNEUMONIA":
        st.warning("AI detects patterns of Pneumonia. Please consult a doctor.")
    else:
        st.success("The X-ray appears Normal.")
