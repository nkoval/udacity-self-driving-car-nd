import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import csv
import json
import cv2

tf.python.control_flow_ops = tf

data_dir = './test_data/'

# Input images format
ch, row, col = 3, 160, 320


# Load the CSV file with the data
def load_csv(file_name):
    data = csv.reader(open(file_name), delimiter=",", quotechar='|')

    return list(data)

csv_data = load_csv(data_dir + 'driving_log.csv')


# Parse the CSV file and get images and angles
def load_data(csv):
    data_len = len(csv)-1
    images = np.empty((data_len, row, col, ch), dtype='uint8')
    angles = np.empty(data_len)

    for i in range(1, data_len):
        img_src = csv[i][0]
        images[i] = mpimg.imread(img_src)
        angles[i] = float(csv[i][3])

    return images, angles

print("Load images...")
X_train, y_train = load_data(csv_data)
print(len(X_train), "images loaded")


# Convert to YUV color space and normalize image
def process_images(images):
    processed = rgb_to_yuv(images)
    processed = [normalize_images(image) for image in processed]
    return np.asarray(processed)


# Normalize an image
def normalize_images(image_data):
    return (image_data / 127.5) - 1.


# Convert RGB to YUV
def rgb_to_yuv(images):
    hsv_images = [cv2.cvtColor(img, cv2.COLOR_RGB2YUV) for img in images]
    return np.asarray(hsv_images)


# We don't need space precision
# Round to 2 decimal places
def normalize_angles(angles):
    return (angles * 100).astype(dtype=int)


# Enhance the data set by copping and flipping images that have steering angle more than `angle_threshold`
# In this way we add more turns to our training data set
def enhance_data(images, angles, angle_threshold):
    indices = np.argwhere(abs(angles) >= angle_threshold * 100)
    indices = np.squeeze(indices)

    angles_to_flip = angles[indices]
    reversed_angles = flip_angles(angles_to_flip)

    images_to_flip = images[indices]
    flipped_images = flip_images(images_to_flip)

    return np.append(images, flipped_images, axis=0), np.append(angles, reversed_angles, axis=0)


# Flips the images in horizontal direction
def flip_images(images_to_flip):
    return np.asarray([np.fliplr(img) for img in images_to_flip])


# Flips the angles
def flip_angles(angles_to_flip):
    return -angles_to_flip

# Round angles to 2 decimal places
y_normalized = normalize_angles(y_train)

# Normalize and convert images to YUV color scheme
X_processed = process_images(X_train)

# Augment data set by adding mirrored turns to the original data set
X_processed, y_normalized = enhance_data(X_processed, y_normalized, 0.1)

print(len(X_processed), 'images after enhancing')

X_processed, y_normalized = shuffle(X_processed, y_normalized)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam


model = Sequential()

model.add(Cropping2D(cropping=((1, 1), (25, 25)), input_shape=(row, col, ch)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(16, 8, 8, border_mode='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 5, 5, border_mode='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Activation('elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

EPOCHS = 10
BATCH_SIZE = 64

history = model.fit(X_processed, y_normalized, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.1)

model.save_weights("./model.h5", True)
with open('./model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
