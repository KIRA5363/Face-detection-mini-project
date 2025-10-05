# streamlit_demo.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Custom CSS for hacker style
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        color: #00FF00;
        font-family: 'Courier New', Courier, monospace;
    }
    .css-1d391kg {
        background-color: #000000;
    }
    .stButton>button {
        background-color: #003300;
        color: #00FF00;
        font-family: 'Courier New', Courier, monospace;
        border: 1px solid #00FF00;
    }
    .stFileUploader>div>div>input {
        background-color: #000000;
        color: #00FF00;
        font-family: 'Courier New', Courier, monospace;
    }
    .stImage>div>img {
        border: 2px solid #00FF00;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
model = tf.keras.models.load_model("emotion_masked_cnn_model.h5")
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Title
st.title("😷 Masked Face Emotion Recognition")
st.write("Upload a grayscale face image (48x48) to predict emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((48, 48))
    img_array = np.array(image).reshape(1, 48, 48, 1) / 255.0

    st.image(image, caption="Uploaded Image", use_column_width=False)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader(f"Predicted Emotion: {predicted_class.capitalize()} 😐")
    st.write(f"Confidence: {confidence:.2f}")
else:
    st.info("Please upload a grayscale 48x48 face image.")
