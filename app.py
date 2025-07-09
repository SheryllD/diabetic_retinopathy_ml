#!/usr/bin/env python
# coding: utf-8
# # Libraries

import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_keras_model():
    return load_model("CNN_model.keras", compile=False)

model = load_keras_model()

# Predict class from a random image in a random class folder
def predict_class_random_from_dir():
    base_path = "test_images"
    class_folders = ["DR", "No_DR"]
    selected_class = random.choice(class_folders)
    directory_path = os.path.join(base_path, selected_class)

    if not os.path.exists(directory_path):
        st.error(f"Directory not found: {directory_path}")
        return

    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    if not image_files:
        st.warning(f"No image files in {directory_path}")
        return

    selected_image_file = random.choice(image_files)
    path = os.path.join(directory_path, selected_image_file)

    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return

    st.image(img, caption=f"Selected Image: {selected_image_file}", use_column_width=True)

    img = img.resize((224, 224))
    image_array = np.array(img) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predicting
    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction, axis=1)[0]

    # Mapping true and predicted labels
    true_class = os.path.basename(directory_path)
    predicted_class = "Diabetic Retinopathy Detected" if predicted_index == 0 else "No Diabetic Retinopathy Detected"
    true_class_label = "Diabetic Retinopathy Detected" if true_class == "DR" else "No Diabetic Retinopathy Detected"

    if predicted_class == true_class_label:
        st.success("Assessment Accuracy: Diagnosis matches!")
    else:
        st.error("Diagnosis does not match the reference label.")


    st.write(f"**Model Assessment:** {predicted_class}")
    st.write(f"**True Condition:** {selected_class}")

    

# Streamlit interface
st.title("Diabetic Retinopathy Detection")
st.write("Click the button below to randomly choose an image to detect Diabtic Retinopathy")

if st.button("Run Diagnose"):
    predict_class_random_from_dir()
