import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("../models/sickle_model.h5")

st.title("🩸 Sickle Cell Disease Image Classifier")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = image.resize((224, 224))  # adjust size based on your model's input size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    if prediction[0] > 0.5:
        st.write("**Positive for Sickle Cell Disease**")
    else:
        st.write("**Negative for Sickle Cell Disease**")
