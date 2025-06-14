#!/usr/bin/env python
# coding: utf-8

# ## The goal of this is to extract the names of all files in the folder with the required class label. 
# ## Their are lots of images without the needed class label which can affect training efficiency
# ## Because each pixel in all images in the folder needs to be examined, a gpu is needed to speed up operations
# ## Without gpu it could take more than a day
# ## With gpu it can take some minutes/an hour

# ## import library

import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import load_img, img_to_array, array_to_img, to_categorical
import tensorflow as tf
from argparse import ArgumentParser
import gc


# data could have varying size of images, so we use the standard of 1080 by 1920
img_breadth = 1080
img_length = 1920
    

def is_subarray_present(array_2d, array_1d):
    """A tensorflow based function for checking if a 1 d array (which represents a specific label class pixel values) 
       is present in a 2 d array (which represents an image reshaped from 3d to 2d, with the row and coloumn combined as one dimension 
       leaving behind the pixel's RGB values as the second dimension, which is compared to a given label class value)  
    """
    answer = tf.reduce_any(tf.reduce_all(tf.equal(array_2d, array_1d), axis=1))
    return answer.numpy()


def get_needed_name_of_image_with_label(label1, label2, input_dir):
    """ A function that goes to a folder, takes a label, and check if a given label class (or some label classes) are present in the label;
        after which the function returns the name of all labels in the folder with the desired label class (or classes)
    """
    
    print( "the gpu device you are using is")
    print(tf.config.list_physical_devices("GPU"))
    print("\n")
    
    gc.collect()
    label1 = tf.convert_to_tensor(label1)
    label2 = tf.convert_to_tensor(label2)
    
    address = []
    input_img_paths = [os.path.join(input_dir, fname) 
         for fname in os.listdir(input_dir) 
             if fname.endswith('png')]

    for x in input_img_paths:
        image_x = img_to_array(load_img(x, target_size=(img_breadth, img_length) ))
        img_x_adjust = image_x.reshape((np.size(image_x,0)*np.size(image_x,1),3))

        img_x_adjust = tf.convert_to_tensor(img_x_adjust)

        label_present1 = is_subarray_present(img_x_adjust, label1)
        label_present2 = is_subarray_present(img_x_adjust, label2)
        
        if ( (label_present1 == True) or (label_present1 == True) ):
            address.append(x)
        else:
            pass
       
    return address
    

def remove_directory_file_type_from_name(x, dir):
    """This function takes the name of a label and strip out the directory and .png component,
       so that if the names are stored in a  csv file, the csv file can be used in another system with a different directory name
    """
    x1 = x.split(dir)
    x2 = '/' + x1[1]
    x3 = x2.split(".png")
    return x3[0] 


def main():
    parser = ArgumentParser()
    parser.add_argument("-tr", "--trainlabels", dest="train_labels_filepath", help="path to folder with train ground truth labels")
    parser.add_argument("-va", "--valilabels", dest="val_labels_filepath", help="path to folder with validation ground truth labels")
    args = parser.parse_args()
    
    all_val_address = args.val_labels_filepath
    all_train_address = args.train_labels_filepath
    
    # ## get val
    print("processing val data in progress")
    needed_val_address = get_needed_name_of_image_with_label([210.0, 60.0, 60.0], [250.0, 170.0, 35.0], all_val_address)
    needed_val_address_modified = [remove_directory_file_type_from_name(x, all_val_address) for x in needed_val_address]
    needed_val_address_modified_pd=pd.DataFrame(needed_val_address_modified, columns=['address'])
    needed_val_address_modified_pd.to_csv('val_address_modified.csv', index = False)
    print("val is completed with {} names extracted".format(len(needed_val_address_modified_pd)))
    
    # ## get val train
    print("processing train data in progress")
    needed_train_address = get_needed_name_of_image_with_label([210.0, 60.0, 60.0], [250.0, 170.0, 35.0], all_train_address)
    needed_train_address_modified = [remove_directory_file_type_from_name(x, all_train_address) for x in needed_train_address]
    needed_train_address_modified_pd=pd.DataFrame(needed_train_address_modified, columns=['address'])
    needed_train_address_modified_pd.to_csv('train_address_modified.csv', index = False)
    print("train is completed with {} names extracted".format(len(needed_train_address_modified_pd)))
    

if __name__ == '__main__':
    main()
