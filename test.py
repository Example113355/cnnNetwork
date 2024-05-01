import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Resizing, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential

# folder
test_folder = 'Data/test'
train_folder = 'Data/train'
val_folder = 'Data/valid'

# Load dataset
IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=1
EPOCH=50

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_folder,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_folder,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_folder,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)   

# Define CNN architecture
model = Sequential()

model.add(Resizing(IMAGE_SIZE, IMAGE_SIZE))
model.add(Rescaling(1./255))
model.add(RandomFlip('horizontal_and_vertical'))
model.add(RandomRotation(0.2))

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile and fit the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

pickle.dump(model, open('model.pkl', 'wb'))

