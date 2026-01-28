# ==========================================
# STEP 1: SETUP KAGGLE & DOWNLOAD DATA
# ==========================================
import os

# Set your Kaggle API details (Using your active token)
os.environ['KAGGLE_USERNAME'] = "emanshahid"
os.environ['KAGGLE_KEY'] = "KGAT_1ffa0be4a0e7ca68b29a5fe9ffdba3fc"

# Download the Pneumonia Dataset from Kaggle
print("Step 1: Downloading dataset...")
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Unzip the downloaded files into a folder named 'dataset'
print("Step 2: Extracting files...")
!unzip -q chest-xray-pneumonia.zip -d /content/dataset/
print("Data is ready!")

# ==========================================
# STEP 2: PREPARE DATA (DATA AUGMENTATION)
# ==========================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to the folders
train_dir = '/content/dataset/chest_xray/train'
test_dir = '/content/dataset/chest_xray/test'
val_dir = '/content/dataset/chest_xray/val'

# Rescale images (Divide by 255) to make values between 0 and 1
# Also add 'zoom' and 'flip' to help the model learn better (Augmentation)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from folders in batches of 32
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
val_generator = test_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

# ==========================================
# STEP 3: CREATE THE CNN MODEL (THE BRAIN)
# ==========================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# Layer 1: Find simple patterns (edges, lines)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2)) # Shrink image size

# Layer 2: Find more complex patterns
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Layer 3: Deep features
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Flatten: Convert 2D image data into 1D list
model.add(Flatten())

# Dense Layer: Final decision making
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # Prevent the model from memorizing (Overfitting)
model.add(Dense(1, activation='sigmoid')) # Output: 0 (Normal) or 1 (Pneumonia)

# Compile: Set the rules for learning
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# STEP 4: TRAIN THE MODEL
# ==========================================
print("Step 3: Starting Training...")
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# ==========================================
# STEP 5: VISUALIZE RESULTS (GRAPHS)
# ==========================================
import matplotlib.pyplot as plt

# Create Accuracy and Loss plots
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# ==========================================
# STEP 6: PREDICTION & PROMPT ENGINEERING
# ==========================================
from google.colab import files
from tensorflow.keras.preprocessing import image
import numpy as np

# Upload an image to test
uploaded = files.upload()

for filename in uploaded.keys():
    # Load and process the image
    img = image.load_img(filename, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get Prediction
    pred = model.predict(img_array)
    status = "PNEUMONIA" if pred[0][0] > 0.5 else "NORMAL"
    confidence = pred[0][0] if status == "PNEUMONIA" else 1 - pred[0][0]

    # PROMPT ENGINEERING: Generating a Human-Readable Report
    prompt_report = f"""
    --- AI SCREENING REPORT ---
    The CNN Model analyzed the X-ray image.
    RESULT: {status}
    CONFIDENCE SCORE: {confidence*100:.2f}%

    ADVICE:
    1. If confidence is below 80%, please double-check with a Senior Doctor.
    2. Maintain hydration and monitor breathing patterns.
    3. DISCLAIMER: This is an AI-generated result and not a final medical diagnosis.
    """
    print(prompt_report)

# Save the model for Deployment later
model.save('pneumonia_model.h5')
print("Model saved as pneumonia_model.h5")