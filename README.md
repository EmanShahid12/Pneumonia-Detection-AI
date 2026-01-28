 AI Pneumonia Diagnosis & Smart Reporting System
This project is an end-to-end Medical Imaging solution that uses Deep Learning to detect Pneumonia from Chest X-ray images and provides an AI-generated medical summary using Prompt Engineering principles.

 Project Overview
Pneumonia is a significant health challenge, and early detection via X-rays is crucial. This system automates the screening process and bridges the gap between raw data and patient understanding by generating a structured report.

 Technical Stack
Deep Learning Framework: TensorFlow / Keras

Model Architecture: Convolutional Neural Network (CNN)

Data Source: Kaggle Chest X-ray Dataset (Pneumonia)

Language: Python

Reporting: Prompt Engineering logic for structured AI feedback

 Model Performance
The model was trained for 10 epochs, achieving high accuracy on the training set.

Training Accuracy: ~95%

Validation: Monitored via Loss/Accuracy graphs to ensure robustness.

 Key Features
Automated Classification: Distinguishes between 'Normal' and 'Pneumonia' X-rays with a confidence score.

Smart Advice: Uses prompt-based logic to suggest next steps (e.g., "Consult a radiologist if confidence is low").

Data Augmentation: Used ImageDataGenerator to make the model robust against different image angles and zooms.

 Project Structure
Untitled0.ipynb: Complete code for data loading, training, and testing.

pneumonia_model.h5: The pre-trained weights of the CNN model.

requirements.txt: List of necessary Python libraries for deployment.

 Disclaimer

This tool is for educational and screening purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
https://drive.google.com/file/d/128vMTkE6K2mNvNrat-xdjO9YIwTVgxe8/view?usp=drive_link
