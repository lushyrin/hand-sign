import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# def load_and_preprocess_image(path):
#     print('cp0')
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
#     image = cv2.resize(image, (200, 200))  # Resize to fit the model input
#     image = image / 255.0  # Normalize pixel values
#     return image

# # Set up directories
# base_dir = 'asl_dataset'
# # categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# categories = ['A', 'B', 'C']

# # ImageDataGenerator for data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.2,
#     horizontal_flip=False,
#     fill_mode='nearest'
# )

# # Assume we have a directory for each category
# for category in categories:
#     path = os.path.join(base_dir, category)
#     images = [os.path.join(path, img) for img in os.listdir(path)]
#     for img_path in images:
#         img = load_and_preprocess_image(img_path)
#         img = img.reshape((1,) + img.shape + (1,))  # Reshape for data generator
#         # Generate and save augmented images
#         i = 0
#         for batch in datagen.flow(img, batch_size=1, save_to_dir=path, save_prefix='aug', save_format='png'):
#             i += 1
#             if i > 5:  # Generate 5 new images per original image
#                 break


categories = ['A', 'B', 'C']

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(200, 200, 1)),  # Input layer specifying the input shape
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')  # Output layer for 26 letters
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'asl_dataset',
    target_size=(200, 200),
    batch_size=20,
    color_mode='grayscale',
    class_mode='sparse')

history = model.fit(train_generator, epochs=10)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 200, 200, 1)
    prediction = model.predict(reshaped)
    sign_index = np.argmax(prediction)
    sign_name = categories[sign_index]

    cv2.imshow('Sign Language Translator', frame)
    print("Predicted Sign:", sign_name)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
