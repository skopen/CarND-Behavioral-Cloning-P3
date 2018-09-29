import csv
import cv2
from scipy import ndimage
import numpy as np

lines = []
isHeader = True

with open ('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if isHeader == False:
            lines.append(line)
        else:
            isHeader = False
images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    # image = cv2.imread(current_path)
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float (line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle=True)

model.save('model.h5')