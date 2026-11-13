import streamlit as st
import os
import numpy as np
from PIL import Image

st.set_page_config(page_title="Animal Species Prediction", page_icon="🐯")

st.title("🐯 Animal Species Prediction")
st.markdown("Project 4: VGG-16 | Animals-10 Dataset | TensorFlow")

# Check TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing.image import img_to_array
    TF_AVAILABLE = True
    st.success(f"TensorFlow {tf.__version__} Loaded | Python 3.11 Ready")
except Exception as e:
    TF_AVAILABLE = False
    st.error(f"TensorFlow install karo: py -3.11 -m pip install tensorflow==2.15.0 Error: {e}")
    st.stop()

# Load VGG16 - Pre-trained on ImageNet (14 Million Images)
@st.cache_resource
def load_model():
    model = VGG16(weights='imagenet')
    return model

with st.spinner("VGG-16 Model Load ho raha hai (Pehli dafa 500MB download hoga)..."):
    model = load_model()

# Animals-10 Classes
ANIMAL_CLASSES = ['cat', 'dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cow', 'sheep', 'squirrel', 'spider']

st.sidebar.header("Project Info")
st.sidebar.write("**Dataset:** Animals-10 Kaggle **Model:** VGG-16 Pre-trained on ImageNet (14 Million Images) **Tools:** TensorFlow, Keras, Python **Type:** Multi-class Classification")

uploaded_file = st.file_uploader("Animal ki image upload karo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=5)[0]
    
    st.subheader("Prediction Result:")
    for i, (imagenet_id, label, score) in enumerate(decoded):
        st.write(f"**{i+1}. {label}**: {score*100:.2f}%")
    
    # Best match in Animals-10
    top_label = decoded[0][1]
    st.success(f"Ye animal lag raha hai: **{top_label}**")

else:
    st.info("Image upload karo prediction ke liye")