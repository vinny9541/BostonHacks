import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load the trained model
model = load_model('CNN_model')  # Replace with the path to your saved model

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
    if prediction[0] > 0.5:
        return "Recycling"
    else:
        return "Trash"

# Streamlit app
st.title('Boston Hacks')
st.header('Trash or Recycle')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict and display the classification
    preprocessed_image = preprocess_image(image, 150, 150)  # Use the same size as during training
    result = predict_image_class(preprocessed_image, model)
    st.write(f"The image is classified as: {result}")
