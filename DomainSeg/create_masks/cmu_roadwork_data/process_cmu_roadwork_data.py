#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import random
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from tensorflow.keras.preprocessing import image
from argparse import ArgumentParser
from PIL import Image


def modify_address(x, val_or_train):
        x1 = x.split("sem_seg_labels/gtFine/"+val_or_train+"/")
        x2 = x1[1].split("_Ids")
        y = x1[0] + 'images/' + x2[0] + '.jpg'
        return y


def shuffle_directories(train_input_dir, filt_type, seed, shuffle, chunkstart, chunkend, val_or_train, is_it_label_address):
    train_input_img_paths = [os.path.join(train_input_dir, fname) 
         for fname in os.listdir(train_input_dir) 
             if fname.endswith(filt_type)]

    if (is_it_label_address==False):
        train_input_img_paths = [modify_address(x, val_or_train) for x in train_input_img_paths]
    else:
        pass
        
    if shuffle == True:
        random.Random(seed).shuffle(train_input_img_paths)
    else:
        pass
    return train_input_img_paths[chunkstart : chunkend]

    
def convert_color_to_mask_class (x, img_breadth, img_length, label_list ): 
    y = np.zeros((img_breadth, img_length, 1))
    for label in label_list:
        y[:,:,0] = np.where(  ((x[:,:,0]==label[0])   & (x[:,:,1]==label[1])   & (x[:,:,2]==label[2]))  , 1 , y[:,:,0])
    return y

def path_to_target (path, img_breadth, img_length, label):
    mask = img_to_array(load_img(path, target_size=(img_breadth, img_length)  ))
    mask = convert_color_to_mask_class (mask, img_breadth, img_length, label)
    return mask


def main():
    
    parser = ArgumentParser()
    parser.add_argument("-tr", "--trainlabels", dest="train_labels_filepath", help="path to folder with train ground truth labels")
    parser.add_argument("-va", "--valilabels", dest="val_labels_filepath", help="path to folder with validation ground truth labels")
    parser.add_argument("-lbs", "--labels-save", dest="labels_save_path", help="path to folder where processed labels will be saved")
    parser.add_argument("-ims", "--images-save", dest="images_save_path", help="path to folder where corresponding images will be saved")
    args = parser.parse_args()

    address_train_label = shuffle_directories(args.train_labels_filepath, '_Ids.png', 0, False, 0, None, 'train', True)  
    address_train_image = shuffle_directories(args.train_labels_filepath, '_Ids.png', 0, False, 0, None, 'train', False)
    address_val_label = shuffle_directories(args.val_labels_filepath, '_Ids.png', 0, False, 0, None, 'val', True)   
    address_val_image = shuffle_directories(args.val_labels_filepath, '_Ids.png', 0, False, 0, None, 'val', False)

    number_of_train = len(address_train_label)
    number_of_val = len(address_val_label)
    
    # get default shape of label
    label_o = Image.open(str(address_train_label[0]))
    img_length = label_o.size[0]
    img_breadth = label_o.size[1]

    label_list = []
    for i in range(6, 9, 1):
        label_list.append( (float(i), float(i), float(i)) )
    for i in range(10, 20, 1):
        label_list.append( (float(i), float(i), float(i)) )

    ## create folder to save image and label
    folder_path = args.images_save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    folder_path = args.labels_save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


    # process training image
    for index in range(0, number_of_train):
        # Open images and pre-existing masks
        image = Image.open(str(address_train_image[index]))
        label = address_train_label[index]
        
        # Create new Coarse Segmentation mask
        coarseSegColorMap = path_to_target(label, img_breadth, img_length, label_list )
        
        # Save images
        image.save(args.images_save_path +'/' + str(index) + ".jpg")
        array_to_img(coarseSegColorMap).save(args.labels_save_path + '/' + str(index) + ".png")
        print(f'Processing training image {index} of {number_of_train-1}')
        
    print('----- Processing training complete -----') 
    
    # process val image
    for index in range(0, number_of_val):
        # Open images and pre-existing masks
        image = Image.open(str(address_val_image[index]))
        label = address_val_label[index]
        
        # Create new Coarse Segmentation mask
        coarseSegColorMap = path_to_target(label, img_breadth, img_length, label_list )
        index = index + number_of_train
        # Save images
        image.save(args.images_save_path + '/' + str(index) + ".jpg")
        array_to_img(coarseSegColorMap).save(args.labels_save_path + '/' + str(index) + ".png")
        print(f'Processing validation image {index - number_of_train} of {number_of_val-1}')
        
    print('----- Processing validation complete -----') 



if __name__ == '__main__':
    main()
