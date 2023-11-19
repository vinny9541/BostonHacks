import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Define paths to datasets
train_dir = 'train'
val_dir = 'validation'
test_dir = 'test'

# Define image dimensions
img_width, img_height = 150, 150

# Batch size
batch_size = 32

# Create data generators with augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescale for the validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    val_dir,  # Updated to new validation directory
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Create a new data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,  # Updated to new test directory
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# train model
epochs = 15

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# evaluate model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

model.save('CNN_model.keras')

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


test_generator.reset()  # Reset generator to start from the beginning
predictions = model.predict(test_generator)

# Display some images and their predictions
num_images_to_display = 100
for i in range(num_images_to_display):
    image, true_label = test_generator.next()
    predicted_label = predictions[i]

    # Convert to binary (0 or 1)
    true_label = int(true_label[0])
    predicted_label = 1 if predicted_label > 0.5 else 0

    # Display the image along with true and predicted labels
    plt.imshow(image[0])
    plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
    plt.show()