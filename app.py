import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
import gdown

# Download model from Google Drive (if needed)
MODEL_PATH = 'model/bone_fracture_classifier.h5'
MODEL_FILE_ID = '1TWtFYwtdNDflB93LAkhkCFoBh6G0_vhI'

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_FILE_ID}', MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# Load the model
model = load_model(MODEL_PATH)

# Function to predict the class of an uploaded image
def predict_fracture(image, model):
    image = image.resize((224, 224))  # Resize image to match the input size of the model
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Map the predicted class to fracture categories
    fracture_categories = ['Spiral Fracture', 'Pathological Fracture', 'Oblique Fracture', 'Longitudinal Fracture', 'Impacted Fracture', 'Hairline Fracture', 'Greenstick Fracture','Fracture Dislocation', 'Comminuted Fracture', 'Avulsion Fracture']
    return fracture_categories[predicted_class[0]]

# Streamlit UI
st.title("Bone Fracture Classification")
st.write("Upload an X-ray image to classify the bone fracture.")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    result = predict_fracture(image, model)
    st.success(f'Predicted Fracture Category: {result}')
