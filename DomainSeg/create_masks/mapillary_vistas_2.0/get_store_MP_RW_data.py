#!/usr/bin/env python
# coding: utf-8

# ## import libraries

import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from argparse import ArgumentParser


# csv files containing names of required images and labels ( csv files should be in same folder as this .py script)
csv_file_train = "train_address_modified.csv"
csv_file_val = "val_address_modified.csv"

# mapillary vistas data have varying size of images, so we use the standard of 1080 by 1920
img_breadth = 1080
img_length = 1920
    

def get_real_address_from_csv(dir_label, dir_image, csv_file):
    """A csv file in the same folder as this .py file would have names of required images & labels to be extracted and preprocessed.
       The csv file is a direct output of the script "extract_relevant_file_names_MP.py".
       This function generates a list of image names and label names using the names of the directories 
       where we have the original Mapillary Vistas images and labels in our computer
    """
    pandas_address = pd.read_csv(csv_file)
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

    # ## read and display image , label and class focused label i for train
    [train_image_address_list, train_label_address_list] = get_real_address_from_csv(dir_train_label, dir_train_image, csv_file_train)
    number_of_train = len(train_image_address_list)

    # ## read and display image , label and class focused label i for val
    [val_image_address_list, val_label_address_list] = get_real_address_from_csv(dir_val_label, dir_val_image, csv_file_val)
    number_of_val = len(val_image_address_list)

    # desired label class
    labelA = [210, 60, 60] 
    labelB = [250, 170, 35]
    label_list = [labelA, labelB]

    ## create folder to save image and label
    folder_path = 'image' # args.images_save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    folder_path = 'label' # args.labels_save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    
    # process training image
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

    # process val image
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
