import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


categories = ['A', 'B', 'C']

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(200, 200, 1)),  
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
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
