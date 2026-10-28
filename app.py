import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="centered"
)

# Application title
st.title("🩺 Skin Disease Detection")
st.write(
    "Upload a skin image to get an AI-based image classification result."
)

st.warning(
    "Educational project only. This application cannot diagnose "
    "skin diseases. Consult a qualified dermatologist for medical advice."
)

# Model configuration
MODEL_PATH = "model/skin_model.keras"

# Replace these categories with the exact class names
# and order used when training your model.
CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevus",
    "Vascular Lesion"
]

# Load trained model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


# Image upload
uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Your uploaded image", use_container_width=True)

    if st.button("Analyze Image"):

        try:
            model = load_model()

            if model is None:
                st.error(
                    "Trained model not found. Please place "
                    "'skin_model.keras' inside the model folder."
                )
                st.stop()

            # Convert image to NumPy array
            image_array = np.array(image)

            # Resize image to match the model input size
            image_array = cv2.resize(image_array, (224, 224))

            # Normalize pixel values
            image_array = image_array.astype("float32") / 255.0

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # Model prediction
            predictions = model.predict(image_array, verbose=0)
            probabilities = predictions[0]

            # Find predicted class
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index]) * 100

            # Display result
            st.subheader("Classification Result")
            st.write(
                f"**Predicted category:** "
                f"{CLASS_NAMES[predicted_index]}"
            )
            st.write(f"**Model confidence:** {confidence:.2f}%")

            st.info(
                "This is an experimental model output, not a medical "
                "diagnosis. Model confidence does not indicate the "
                "probability that you have a disease."
            )

        except Exception as error:
            st.error(f"Unable to analyze the image: {error}")

else:
    st.info("Please upload a skin image to get started.")

st.divider()
st.caption("Python | OpenCV | TensorFlow | Streamlit")
