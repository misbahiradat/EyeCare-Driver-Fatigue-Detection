# Author @ Misbah Iradat

# import libraries

# importing libraries
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

plt.style.use('fivethirtyeight')



def img_generator(WIDTH, HEIGHT,batch_size,pred_path):

    """ This function takes in 4 arguments from the user:
    1. WIDTH : width of the image in pixel
    2. HEIGHT : height of the image in pixel
    (make sure both WIDTH and HEIGHT are same as used in the training!
    3. batch_size: this will be all the images to be predicted (number of images in the folder for prediction)
    4. pred_path: path where the images are stored"""

    pred_datagen = ImageDataGenerator(rescale=1./255)
    pred_generator = pred_datagen.flow_from_directory(
    directory= pred_path,
    target_size=(WIDTH, HEIGHT),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
    seed=42)

    return pred_generator

def make_grid_plot(generator, batch_size,path_model):

    ncols = 3
    nrows = round(int(batch_size/ncols))+1
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)
    fig.subplots_adjust(hspace = .05, wspace=0.9)
    x_pred = generator.__next__()
    model = load_model(path_model, compile=False)

    for i in range(0,batch_size):
        predict=model.predict(generator, verbose = 0)
        y_classes = predict.argmax(axis=-1)
    
        if y_classes[i] == 0:
            label = 'Normal_Eye'
        else:
            label = 'Red_Eye'
        for j in range(0,batch_size):
        
            ax = fig.add_subplot(ncols,nrows,i+1)
            ax.set_title(label)
            image = x_pred[i]
            #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.subplot(ncols,nrows,i+1)
        plt.imshow(image)
        
    plt.show()

if __name__ == "__main__":

    # image parameters:

    WIDTH = 128
    HEIGHT = 128
    CHANNELS = 3
    CLASSES = 2
    path_model = '../Model_Outputs/2022_08_23/exp_imb_data_200ep/model/model.h5'
    pred_path = '../../data/aug_red_eye/prediction_images'
    number_of_images = '../../data/aug_red_eye/prediction_images/prediction'
    img_list_len = len(os.listdir(number_of_images))
    batch_size = int(img_list_len)

    pred_generator = img_generator(WIDTH, HEIGHT,batch_size,pred_path)
    make_grid_plot(pred_generator, batch_size,path_model)