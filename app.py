import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# --------------------- Download model if not exists ---------------------
model_path = 'chilipowder_model.h5'
if not os.path.exists(model_path):
    with st.spinner("📦 Downloading model..."):
        file_id = '1Y-01Q5VtFFkPdNrwOiY_cyHR2zowC8qh'
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        gdown.download(url, model_path, quiet=False)

# --------------------- Load model ---------------------
model = tf.keras.models.load_model(model_path)
class_names = ["100-0", "80-20", "70-30", "60-40", "50-50"]

# --------------------- Page config ---------------------

st.set_page_config(page_title="Chili Powder Classifier", page_icon="🌶️", layout="centered")


# --------------------- Sidebar ---------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909765.png", width=120)
st.sidebar.title("🌶️ Navigation")
st.sidebar.markdown("""
**Steps to use:**
1. Upload an image of chili powder
2. Wait for the model to predict
3. See result and confidence

---

📌 *Supports JPEG, JPG, PNG*
""")

# --------------------- Header ---------------------
st.markdown("""
    <div style='text-align:center'>
        <h1 style='background: linear-gradient(to right, #ff4e50, #f9d423);
                   -webkit-background-clip: text;
                   color: transparent;
                   font-size: 3em;'>Chili Powder Fault Detection 🌶️</h1>
        <p style='font-size: 18px;'>Detect the purity ratio of chili powder using deep learning</p>
    </div>
""", unsafe_allow_html=True)

# --------------------- File Upload ---------------------
uploaded_file = st.file_uploader("📤 Upload a chili powder image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner('Analyzing...'):
        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

    # Display result
    st.success(f"✅ **Prediction:** `{pred_class}`")
    st.info(f"🔍 **Confidence:** `{confidence:.2f}%`")
else:
    st.warning("📁 Please upload an image to get started.")

# --------------------- Footer ---------------------
st.markdown("""
<hr>
<div style='text-align: center; font-size: 16px;'>
    Made with ❤️ by Akash Ghosh · MCA Project 2025
</div>
""", unsafe_allow_html=True)
