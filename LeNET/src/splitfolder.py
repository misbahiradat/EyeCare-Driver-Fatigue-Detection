# Author @ Misbah Iradat
# import libraries
import splitfolders as sf

# path
FOLDER_NAME = '../../data/eye_data_main/'

# splitfolder
sf.ratio(FOLDER_NAME, output="../../data/aug_data_imb", seed=142, ratio=(0.7,0.25,0.05))