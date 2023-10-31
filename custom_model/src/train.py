# Author @ Misbah Iradat
# import libraries

from model import Custom_Model
# importing libraries
import os
import pandas as pd
import numpy as np
from datetime import datetime 
from numpy.random import seed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import keras
from keras.initializers import glorot_uniform
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D ,AveragePooling2D

from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from keras import regularizers
from keras.regularizers import l2, l1
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, save_img
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

plt.style.use('fivethirtyeight')

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# function to create all the necessary directory!

def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            #print("New directory is created")


def metricplot(df, xlab, ylab_1,ylab_2, path):
    
    """
    This function plots metric curves and saves it
    to respective folder
    inputs: df : pandas dataframe 
            xlab: x-axis
            ylab_1 : yaxis_1
            ylab_2 : yaxis_2
            path: full path for saving the plot
            """
    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(x = df[xlab], y = df[ylab_1])
    sns.lineplot(x = df[xlab], y = df[ylab_2])
    plt.xlabel('Epochs',fontsize = 12)
    plt.ylabel(ylab_1,fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend([ylab_1,ylab_2], prop={"size":12})
    plt.savefig(path+'/'+ ylab_1)
    #plt.show()

def Confusion_Matrix(data_generator, model, path):
    
    LABELS = ["Normal_Eye","Red_Eye"]
    x_t, y_t = data_generator.__next__()
    y_series_test = []
    # checking/ verifying if the image and masks are coorelated
    for i in range(0,32):
        image = x_t[i]
        y_series_test.append(int(y_t[i][1]))

    y_pred_test = model.predict(data_generator)
    y_pred_test = y_pred_test.argmax(axis=-1)
    cf_matrix = confusion_matrix(y_series_test, y_pred_test)
    
    sns.heatmap(cf_matrix, annot=True, xticklabels=LABELS, yticklabels=LABELS,fmt = 'd')
    plt.title("Confusion_Martix")
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    #plt.show()
    plt.savefig(path+'/'+ 'ConfusionMatrix')
    print(cf_matrix)

if __name__ == '__main__':
     
    seed(42)
    tf.random.set_seed(42) 
    keras.backend.clear_session()

    # image parameters:

    WIDTH = 128
    HEIGHT = 128
    CHANNELS = 3
    CLASSES  = 2

    # model hyperparameters!

    batchsize = 32
    epochs = 200

    # creating main folder
    today = datetime.now()
    today  = today.strftime('%Y_%m_%d')
    path = '../Model_Outputs/'+ today
    create_dir(path)
 
    # creating directory to save model and its output
    EXPERIMENT_NAME = input('Enter new Experiment name:')
    print('\n')
    print('A folder with',EXPERIMENT_NAME,'name has be created to store all the model details!')
    print('\n')
    folder = EXPERIMENT_NAME
    path_main = path + '/'+ folder
    create_dir(path_main)

    # creating directory to save all the metric data
    folder = 'metrics'
    path_metrics = path_main +'/'+ folder
    create_dir(path_metrics)

    # creating folder to save model.h5 file
    folder = 'model'
    path_model = path_main +'/'+ folder
    create_dir(path_model)

    # creating folder to save model.h5 file
    folder = 'model_checkpoint'
    path_checkpoint = path_main +'/'+ folder
    create_dir(path_checkpoint) 
     
# training path:

train_data_path = '../../data/aug_data_imb/training_aug/train'
val_data_path = '../../data/aug_data_imb/training_aug/val'
test_data_path = '../../data/aug_data_imb/training_aug/test'


# defining ImageGenerator

train_datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255,
        fill_mode="nearest")
val_datagen = ImageDataGenerator(
                                rescale=1./255,
                                )
test_datagen = ImageDataGenerator(
                                rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_path,
        color_mode="rgb",
        target_size=(WIDTH, HEIGHT),
        batch_size=batchsize,
        class_mode="categorical",
        subset='training',
        shuffle=True,
        seed=42
        )
validation_generator = val_datagen.flow_from_directory(
        val_data_path,
        color_mode="rgb",
        target_size=(WIDTH, HEIGHT),
        batch_size=batchsize,
        class_mode="categorical",
        subset='training',
        shuffle=True,
        seed=42
        )

test_generator = test_datagen.flow_from_directory(
        test_data_path,
        color_mode="rgb",
        target_size=(WIDTH, HEIGHT),
        batch_size=32,
        shuffle = False,
        class_mode='categorical',
        seed = 42)

print('\n')

# checking/ verifying if the image and labels are coorelated

x_train, y_train = train_generator.__next__()

INPUT_SHAPE = x_train[0].shape
print(INPUT_SHAPE)


# initializing CustomMOdel model and adding first and last layers

model_init = Custom_Model(INPUT_SHAPE, CLASSES)

model = model_init.custom_CNN()
print(model.summary())

# loading weights:
#model.load_weights(path_model+'/'+model_name)

initial_learning_rate = 0.00001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)

# compling the model
model.compile(optimizer=keras.optimizers.Adam(lr_schedule), 
                loss=keras.losses.binary_crossentropy, metrics=['accuracy','Recall','Precision'])

cb = [
    tf.keras.callbacks.ModelCheckpoint(path_checkpoint),
    tf.keras.callbacks.CSVLogger(path_metrics+'/'+'data.csv'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1001, restore_best_weights=False)]

history = model.fit(train_generator,
        batch_size = batchsize, 
        epochs = epochs,
        verbose = 1, 
        validation_data = validation_generator,
        callbacks=[cb])

# save the model

model.save(path_model+'/'+'model.h5')

# reading the data.csv where all the epoch training scores are stored
df = pd.read_csv(path_metrics+'/'+'data.csv')

metricplot(df, 'epoch', 'loss','val_loss', path_metrics)
metricplot(df, 'epoch', 'accuracy','val_accuracy', path_metrics)
metricplot(df, 'epoch', 'recall','val_recall', path_metrics)
metricplot(df, 'epoch', 'precision','val_precision', path_metrics)

print('\n')
print('Model Training Score')
train_loss, train_accuracy, train_recall, train_precision = model.evaluate(train_generator)
print('\n','Evaluation of Training dataset:','\n''\n','train_loss:',round(train_loss,3),'\n','train_accuracy:',round(train_accuracy,3),'\n', 'train_recall:',round(train_recall,3),'\n','train_precision:',round(train_precision,3))
print('\n')

print('\n')
print('Model Validation Score')
val_loss, val_accuracy, val_recall, val_precision = model.evaluate(validation_generator)
print('\n','Evaluation of Validation dataset:','\n''\n','val_loss:',round(val_loss,3),'\n','val_accuracy:',round(val_accuracy,3),'\n', 'val_recall:',round(val_recall,3),'\n','val_precision:',round(val_precision,3))
print('\n')

print('\n')
print('Model Test Score')
test_loss, test_accuracy, test_recall, test_precision = model.evaluate(test_generator)
print('\n','Evaluation of Test dataset:','\n''\n','test_loss:',round(test_loss,3),'\n','test_accuracy:',round(test_accuracy,3),'\n', 'test_recall:',round(test_recall,3),'\n','test_precision:',round(test_precision,3))
print('\n')


# Building Confusion Matrix for the test dataset
print('Confusion Matrix for the Test DataSet:')
print('\n')
Confusion_Matrix(test_generator, model, path_metrics)