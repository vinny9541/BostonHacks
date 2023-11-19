import os
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the trained model
model = load_model('CNN_model.h5') 

# Function to preprocess the image
def preprocess_image(img, img_height, img_width):
    # Ensure the image is in RGB mode, which has 3 channels
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((img_height, img_width))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.  # Rescale the image
    return img_array

# Function to predict the class
def predict_image_class(img_array, model):
    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)

# Function to process images in a directory and evaluate accuracy
def process_and_evaluate(directory_path, img_height, img_width, model):
    actual_classes = []
    predicted_classes = []
    class_mapping = {'trash': 0, 'recycle': 1}  # Define class labels
    for label, class_index in class_mapping.items():
        folder_path = os.path.join(directory_path, label)
        images = os.listdir(folder_path)
        for image_name in images:
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, image_name)
                img = Image.open(image_path)
                img_array = preprocess_image(img, img_height, img_width)
                prediction = predict_image_class(img_array, model)
                predicted_class = int(prediction > 0.5)  # outputs a probability
                actual_classes.append(class_index)
                predicted_classes.append(predicted_class)
    return actual_classes, predicted_classes

# Calculate accuracy and plot the confusion matrix
def plot_confusion_matrix(actual_classes, predicted_classes):
    acc = accuracy_score(actual_classes, predicted_classes)
    print(f'Accuracy: {acc * 100:.2f}%')
    cm = confusion_matrix(actual_classes, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

dataset_directory = 'test'  

actual, predicted = process_and_evaluate(dataset_directory, 150, 150, model)
plot_confusion_matrix(actual, predicted)
