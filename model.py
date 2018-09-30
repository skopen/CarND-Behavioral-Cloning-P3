import csv
from scipy import ndimage
import numpy as np

lines = []
isHeader = True

data_path = './data/'

# read csv metadata first
with open (data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if isHeader == False:
            lines.append(line)
        else:
            isHeader = False
images = []
measurements = []

# read all images and steering angle measurements (and add flipped image for data augmentation)
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = data_path + 'IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float (line[3])
    measurements.append(measurement)

    # add a flipped image
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D

model = Sequential()

# Pre-process (normalize + crop uninteresting parts)
model.add(Lambda (lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Using NVidia architecture
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compile model & fit
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=3)

# Save model
model.save('model.h5')