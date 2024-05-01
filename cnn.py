import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Resizing, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential

test_folder = 'Data/test'
IMAGE_SIZE=256
BATCH_SIZE=32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_folder,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'
)

pickled_model = pickle.load(open('model.pkl', 'rb'))

# Test and evaluate the model
score = pickled_model.evaluate(test_ds)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Predict
# def predict(model):
#     label = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
#     for images_batch, labels_batch in test_ds.take(1):
#         first_image = images_batch[0].numpy().astype('uint8')
#         first_label = labels_batch[0].numpy()

#         batch_prediction = model.predict(images_batch)
#     return label[np.argmax(batch_prediction[0])], label[first_label]

# print(predict(pickled_model))