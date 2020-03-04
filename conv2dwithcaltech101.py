# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


def StridedNet(width, height, depth, classes, reg, init="he_normal"):
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    # if we are using "channels first", update the input shape
    # and channels dimension
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",
                     kernel_initializer=init, kernel_regularizer=reg,
                     input_shape=inputShape))
    # here we stack two CONV layers on top of each other where
    # each layerswill learn a total of 32 (3x3) filters
    model.add(Conv2D(32, (3, 3), padding="same",
                     kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
                     kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.25))

    # stack two more CONV layers, keeping the size of each filter
    # as 3x3 but increasing to 64 total learned filters
    model.add(Conv2D(64, (3, 3), padding="same",
                     kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
                     kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.25))
    # increase the number of filters again, this time to 128
    model.add(Conv2D(128, (3, 3), padding="same",
                     kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
                     kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.25))

    # fully-connected layer
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer=init))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # return the constructed network architecture
    return model


# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import glob
import itertools
import random

LABELS = set(["Faces", "Leopards", "Motorbikes", "airplanes"])

print("[INFO] loading images...")
data = []
labels = []

img_dir_faces = '/home/oto/Downloads/101_ObjectCategories/Faces'
img_dir_leopards = '/home/oto/Downloads/101_ObjectCategories/Leopards'
img_dir_motorbikes = '/home/oto/Downloads/101_ObjectCategories/Motorbikes'
img_dir_airplanes = '/home/oto/Downloads/101_ObjectCategories/airplanes'

image_path_faces = os.path.join(img_dir_faces, '*g')
image_path_leopards = os.path.join(img_dir_leopards, '*g')
image_path_motorbikes = os.path.join(img_dir_motorbikes, '*g')
image_path_airplanes = os.path.join(img_dir_airplanes, '*g')

image_path_faces = glob.glob(image_path_faces)
image_path_leopards = glob.glob(image_path_leopards)
image_path_motorbikes = glob.glob(image_path_motorbikes)
image_path_airplanes = glob.glob(image_path_airplanes)

imagePaths = list(itertools.chain(image_path_faces, image_path_leopards, image_path_motorbikes, image_path_airplanes))

# grab the image paths and randomly shuffle them
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, resize the image to be 32x32 pixels (ignoring
    # aspect ratio), flatten the image into 32x32x3=3072 pixel image
    # into a list, and store the image in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (96, 96))
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, stratify=labels, random_state=42)
# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
opt = Adam(lr=1e-4, decay=1e-4 / 100)
model = StridedNet(width=96, height=96, depth=3,
                         classes=len(lb.classes_), reg=l2(0.0005))
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# train the network
print("[INFO] training network for {} epochs...".format(
    100))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
                        epochs=100)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = 100
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("StridedNet_plot")


