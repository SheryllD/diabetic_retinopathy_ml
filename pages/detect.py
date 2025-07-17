import streamlit as st
from tensorflow.keras.models import load_model
import os
from PIL import Image
import random
import numpy as np


# -------------- MODEL --------------

@st.cache_resource
def load_keras_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "CNN_model.keras")
        st.write("🔍 Attempting to load model from:", model_path)
        model = load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error("Error loading model:")
        st.exception(e)
        return None

model = load_keras_model()

# -------------- RANDOM CHOOSE --------------

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

    # Pick and display image
    selected_image_file = random.choice(image_files)
    image_path = os.path.join(directory_path, selected_image_file)
    img = Image.open(image_path).convert("RGB")
    st.image(img, caption=f"Selected image: {selected_image_file}", use_container_width=True)

    if model is None:
        st.error("Model failed to load. Cannot make predictions.")
        return

    # Preprocess 
    img = img.resize((224, 224))
    x = np.expand_dims(np.array(img) / 255.0, axis=0)

    try:
        prediction = model.predict(x)
        pred_idx = np.argmax(prediction, axis=1)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    pred_class = "DR" if pred_idx == 0 else "No_DR"
    true_class = selected_class

    if pred_class == true_class:
        st.success("Assessment Accuracy: Diagnosis matches!")
    else:
        st.error("Diagnosis does not match the reference label")

    # Display result
    readable_pred = "Diabetic Retinopathy Detected" if pred_class == "DR" else "No Diabetic Retinopathy Detected"
    st.write(f"**Model Assessment:** {readable_pred}")
    st.write(f"**True Condition:** {true_class}")


# -------------- APP INTERFACE --------------

def run_detection():
    st.title("Diabetic Retinopathy Detection")
    st.write("Click the button below to randomly select a retinal image and run the model.")
    st.divider()

    if st.button("Run Diagnose"):
        predict_class_random_from_dir()
