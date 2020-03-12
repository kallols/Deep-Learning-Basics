# USAGE
# python train.py --checkpoints output/checkpoints
# python train.py --checkpoints output/checkpoints \
# 	--model output/checkpoints/epoch_40.hdf5 --start-epoch 40
# python train.py --checkpoints output/checkpoints \
#       --model output/checkpoints/epoch_50.hdf5 --start-epoch 50

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from epochcheckpoint import EpochCheckpoint
from trainingmonitor import TrainingMonitor
from resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.models import load_model
import keras.backend as K
import numpy as np
import cv2
from keras.callbacks import Callback
import os

# grab the Fashion MNIST dataset (if this is your first time running
# this the dataset will be automatically downloaded)
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# Fashion MNIST images are 28x28 but the network we will be training
# is expecting 32x32 images
trainX = np.array([cv2.resize(x, (32, 32)) for x in trainX])
testX = np.array([cv2.resize(x, (32, 32)) for x in testX])

# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# reshape the data matrices to include a channel dimension (required
# for training)
trainX = trainX.reshape((trainX.shape[0], 32, 32, 1))
testX = testX.reshape((testX.shape[0], 32, 32, 1))

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")

# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
model = None
if model is None:
    print("[INFO] compiling model...")
    opt = SGD(lr=1e-1)
    model = ResNet(32, 32, 1, 10, (9, 9, 9),
                         (64, 64, 128, 256), reg=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

# otherwise, we're using a checkpoint model
else:
    # load the checkpoint from disk
    print("[INFO] loading {}...".format(model))
    model = load_model(model)

    # update the learning rate
    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-2)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)))

# build the path to the training plot and training history
plotPath = os.path.sep.join(["/home/oto/PycharmProjects/deep_learning", "resnet_fashion_mnist.png"])
jsonPath = os.path.sep.join(["/home/oto/PycharmProjects/deep_learning", "resnet_fashion_mnist.json"])

# construct the set of callbacks
callbacks = [
    EpochCheckpoint("/home/oto/PycharmProjects/deep_learning", every=5,
                    startAt=0),
    TrainingMonitor(plotPath,
                    jsonPath=jsonPath,
                    startAt=0)]

# train the network
print("[INFO] training network...")
model.fit_generator(
    aug.flow(trainX, trainY, batch_size=128),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // 128,
    epochs=80,
    callbacks=callbacks,
    verbose=1)
