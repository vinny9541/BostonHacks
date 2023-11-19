import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

# Check if the model file exists
model_path = 'CNN_model'  # Replace with the path to your saved model
if not os.path.exists(model_path):
    st.error("Model file not found. Please make sure the path is correct.")
else:
    # Load the trained model
    model = load_model(model_path)

    # function to preprocess the image
    def preprocess_image(img, img_height, img_width):
        img = img.resize((img_height, img_width))
        img_array = tf_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch
        img_array /= 255.  # Rescale the image
        return img_array

    # function to predict the class
    def predict_image_class(img_array, model):
        prediction = model.predict(img_array)
        return prediction[0][0]  # Return the confidence score

# Streamlit app
st.title('Boston Hacks')
st.header('Trash or Recycle')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict and display the classification
        preprocessed_image = preprocess_image(image, 150, 150)  
        confidence_score = predict_image_class(preprocessed_image, model)

        threshold = 0.5
        result = "Recycling" if confidence_score > threshold else "Trash"

        st.write(f"The image is classified as: {result}")
        st.write(f"Confidence Score: {confidence_score:.2%}")  # Display confidence