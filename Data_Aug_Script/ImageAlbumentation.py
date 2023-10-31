# Author @ Misbah Iradat
import numpy as np
from skimage import io
import random
import os
from scipy.ndimage import rotate
from tqdm import tqdm
import itertools


images_path_normal="../data/aug_red_eye/training/test/normal_eye" #path to original images
img_augmented_path_normal="../data/aug_red_eye/training_aug/test/normal_eye_aug/" # path to store aumented images
number_of_images = images_path_normal
img_list_len = len(os.listdir(number_of_images))
images_to_generate = img_list_len
seed_for_random = 42

#Define functions for each operation
# Make sure the order of the spline interpolation is 0, default is 3. 
# With interpolation, the pixel values get messed up.
def rotation(image, seed):
    random.seed(seed)
    angle= random.randint(-180,180)
    r_img = rotate(image, angle, mode='constant', cval = 0, reshape=False, order=0)
    return r_img

def h_flip(image, seed):
    hflipped_img= np.fliplr(image)
    return  hflipped_img

def v_flip(image, seed):
    vflipped_img= np.flipud(image)
    return vflipped_img

def v_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    vtranslated_img = np.roll(image, n_pixels, axis=0)
    return vtranslated_img

def h_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    htranslated_img = np.roll(image, n_pixels, axis=1)
    return htranslated_img


transformations = {'rotate': rotation,
                'horizontal flip': h_flip, 
                'vertical flip': v_flip,
                'vertical shift': v_transl,
                'horizontal shift': h_transl
                 }                #use dictionary to store names of functions 


images_normal=[] # to store paths of images from folder

for im in os.listdir(images_path_normal): 
    images_normal.append(os.path.join(images_path_normal,im))
print(len(images_normal))

for i, im in tqdm(zip(range(images_to_generate), os.listdir(images_path_normal))):

    img_name = im.split('/')[-1].split('.')
    img = img_name[0]
    image = images_normal[i]
    original_image = io.imread(image)
    transformed_image = None

    for key in transformations:

        seed = random.randint(1,100)  #Generate seed to supply transformation functions. 
        transformed_image = transformations[key](original_image, seed)
        new_image_path= img_augmented_path_normal + f'{img}_{key}.png' 
        io.imsave(new_image_path, transformed_image)
        i =i+1