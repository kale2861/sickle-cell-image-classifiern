from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load your model
model = load_model("models/sickle_model.h5")

# Set target size (must match training)
IMAGE_SIZE = (224, 224)

st.title("Sickle Cell Image Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0  # Normalize to 0–1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)[0][0]
    st.write(f"Raw model output: {prediction:.4f}")

    # Interpret result
    if prediction > 0.5:
        st.error("Prediction: Positive for Sickle Cells")
    else:
        st.success("Prediction: Negative for Sickle Cells")
