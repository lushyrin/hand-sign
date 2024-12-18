import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    image = cv2.resize(image, (200, 200))  # Resize to fit the model input
    image = image / 255.0  # Normalize pixel values
    return image

# Set up directories
base_dir = 'asl_dataset'
# categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
categories = ['A', 'B', 'C']
# ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Assume we have a directory for each category
for category in categories:
    path = os.path.join(base_dir, category)
    images = [os.path.join(path, img) for img in os.listdir(path)]
    for img_path in images:
        img = load_and_preprocess_image(img_path)
        img = img.reshape((1,) + img.shape + (1,))  # Reshape for data generator
        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=path, save_prefix='aug', save_format='png'):
            i += 1
            if i > 5:  # Generate 5 new images per original image
                break
