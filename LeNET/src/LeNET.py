# Author @ Misbah Iradat
# importing libraries
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from keras import regularizers
from keras.regularizers import l2, l1


#defining Model

class LeNET_Model():

    def __init__(self, INPUT_SHAPE, CLASSES):

        self.INPUT_SHAPE = INPUT_SHAPE #(WIDTH,HEIGHT,CHANNELS)
        self.CLASSES = CLASSES

    def LeNET(self):

        model = Sequential()

        model.add(Conv2D(filters= 6, kernel_size=(3,3), strides=(1,1), input_shape=self.INPUT_SHAPE,
                activation=keras.activations.relu,
                padding='valid',kernel_regularizer=l2(0.0001)))

        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding= 'valid'))

        model.add(Conv2D(filters = 16, kernel_size=(3,3),strides=(1,1),
                activation=keras.activations.relu, 
                padding = 'valid',kernel_regularizer=l2(0.0001)))

        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding= 'valid'))
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(units=128, activation=keras.activations.relu))
        model.add(BatchNormalization())

        model.add(Dropout(0.3))
        model.add(Dense(units=64, activation=keras.activations.relu))
        model.add(BatchNormalization())

        model.add(Dropout(0.3))
        model.add(Dense(units=32, activation=keras.activations.relu))
        model.add(BatchNormalization())

        model.add(Dense(units = self.CLASSES, activation=keras.activations.sigmoid))

        return model


if __name__ == "__main__":

    WIDTH = 256
    HEIGHT = 256
    CHANNELS = 3
    CLASSES = 2
    INPUT_SHAPE = (128,128,3)
    Model = LeNET_Model(INPUT_SHAPE, CLASSES)
    main_model = Model.LeNET()
    print(main_model.summary())



