#!/usr/bin/env python
# coding: utf-8

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
       so that it can be used for both image and label extraction (or store the names independent of the directory name if desired)
    """
    x1 = x.split(dir)
    x2 = '/' + x1[1]
    x3 = x2.split(".png")
    return x3[0] 


def get_image_and_label_address_from_list(dir_label, dir_image, important_list):
    """A list (important_list) with shortened names of required images & labels to be extracted and preprocessed is provided to this function.
       This function then uses the list to generates a list of real image names and a list of real label names using the names of the directories 
       where we have the original Mapillary Vistas images and labels in our computer
    """
    pandas_address=pd.DataFrame(important_list, columns=['address'])
    pandas_address['real_label_address'] = dir_label[:-1] + pandas_address['address'] + '.png'
    pandas_address['real_image_address'] = dir_image[:-1] + pandas_address['address'] + '.jpg'
    image_address_list = pandas_address['real_image_address'].tolist()
    label_address_list = pandas_address['real_label_address'].tolist()
    return [image_address_list, label_address_list]   


def convert_color_to_mask_class (x, img_breadth, img_length, label_list ):
    """take the original label, create a new label with a class of 1 if the class in the original label is in the label_list, 
       else create a class of 0 in the new label
    """
    y = np.zeros((img_breadth, img_length, 1))
    for label in label_list:
        y[:,:,0] = np.where(  ((x[:,:,0]==label[0])   & (x[:,:,1]==label[1])   & (x[:,:,2]==label[2]))  , 1 , y[:,:,0])
    return y


def path_to_target (path, img_breadth, img_length, label):
    """load original label from directory, create a new label using the convert_color_to_mask_class function
    """
    mask = img_to_array(load_img(path, target_size=(img_breadth, img_length)  ))
    mask = convert_color_to_mask_class (mask, img_breadth, img_length, label)
    return mask

    
def path_to_image (path, img_breadth, img_length ):
    """load image from directory
    """
    return img_to_array(load_img(path, target_size=(img_breadth, img_length) ))




def main():
    parser = ArgumentParser()
    parser.add_argument("-trlb", "--trainlabels", dest="train_labels_filepath", help="path to folder with train ground truth labels")
    parser.add_argument("-trim", "--trainimages", dest="train_images_filepath", help="path to folder with train images")
    parser.add_argument("-valb", "--valilabels", dest="val_labels_filepath", help="path to folder with validation ground truth labels")
    parser.add_argument("-vaim", "--valiimages", dest="val_images_filepath", help="path to folder with validation images")
    parser.add_argument("-lbs", "--labels-save", dest="labels_save_path", help="path to folder where processed labels will be saved")
    parser.add_argument("-ims", "--images-save", dest="images_save_path", help="path to folder where corresponding images will be saved")
    args = parser.parse_args()
    
    dir_train_label = args.train_labels_filepath 
    dir_train_image = args.train_images_filepath 
    dir_val_label = args.val_labels_filepath 
    dir_val_image = args.val_images_filepath 

    ## create folders to save images and labels
    folder_path =  args.images_save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    folder_path =  args.labels_save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    # desired label class should be in float32
    labelA = np.array([210, 60, 60], dtype='float32')
    labelB = np.array([250, 170, 35], dtype='float32')
    label_list = [labelA, labelB]
    

    # ## get train
    print("processing train data in progress")
    needed_train_address = get_needed_name_of_image_with_label(label_list[0], label_list[1], dir_train_label)
    needed_train_address_modified = [remove_directory_file_type_from_name(x, dir_train_label) for x in needed_train_address]
    [train_image_address_list, train_label_address_list] =get_image_and_label_address_from_list(dir_train_label, dir_train_image, needed_train_address_modified)
    number_of_train = len(train_image_address_list)
    
    for index in range(0, number_of_train):
        # Open images and pre-existing masks address
        image = train_image_address_list[index]
        label = train_label_address_list[index]
        # get image
        image = path_to_image (image, img_breadth, img_length )
        # Create new Coarse Segmentation mask
        coarseSegColorMap = path_to_target(label, img_breadth, img_length, label_list )     
        # Save images
        array_to_img(image).save(args.images_save_path +'/' + str(index) + ".jpg")
        # save label
        array_to_img(coarseSegColorMap).save(args.labels_save_path + '/' + str(index) + ".png")
        print(f'Processing training image {index} of {number_of_train-1}')
    print('----- Processing training complete -----') 

    
    
    # ## get val
    print("processing val data in progress")
    needed_val_address = get_needed_name_of_image_with_label(label_list[0], label_list[1], dir_val_label)
    needed_val_address_modified = [remove_directory_file_type_from_name(x, dir_val_label) for x in needed_val_address]
    [val_image_address_list, val_label_address_list] = get_image_and_label_address_from_list(dir_val_label, dir_val_image, needed_val_address_modified)
    number_of_val = len(val_image_address_list)
    
    for index in range(0, number_of_val):
        # Open images and pre-existing masks address
        image = val_image_address_list[index]
        label = val_label_address_list[index]
        # get image
        image = path_to_image (image, img_breadth, img_length )
        # Create new Coarse Segmentation mask
        coarseSegColorMap = path_to_target(label, img_breadth, img_length, label_list )
        index = index + number_of_train
        # Save images
        array_to_img(image).save(args.images_save_path + '/' + str(index) + ".jpg")
        # save label
        array_to_img(coarseSegColorMap).save(args.labels_save_path + '/' + str(index) + ".png")
        print(f'Processing validation image {index - number_of_train} of {number_of_val-1}')
    print('----- Processing validation complete -----') 
    

    
    
    

if __name__ == '__main__':
    main()
