#! /usr/bin/env python3
#%%
import os
import json
import torch
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
)))

from argparse import ArgumentParser
from PIL import Image
from typing import Literal, get_args
from Models.data_utils.load_data_ego_path import LoadDataEgoPath
from Models.training.ego_path_trainer import EgoPathTrainer


def main():

    # ====================== Loading Data ====================== #
    
    # ROOT PATH
    root = '/mnt/media/EgoPath/EgoPathDatasets/'
    
    # BDD100K
    bdd100k_labels_filepath= root + 'BDD100K/drivable_path.json'
    bdd100k_images_filepath = root + 'BDD100K/image/'

    # COMMA2K19
    comma2k19_labels_fileapath = root + 'COMMA2K19/drivable_path.json'
    comma2k19_images_fileapath = root + 'COMMA2K19/image/'

    # CULANE
    culane_labels_fileapath = root + 'CULANE/drivable_path.json'
    culane_images_fileapath = root + 'CULANE/image/'

    # CURVELANES
    curvelanes_labels_fileapath = root + 'CURVELANES/drivable_path.json'
    curvelanes_images_fileapath = root + 'CURVELANES/image/'

    # ROADWORK
    roadwork_labels_fileapath = root + 'ROADWORK/drivable_path.json'
    roadwork_images_fileapath = root + 'ROADWORK/image/'

    # TUSIMPLE
    tusimple_labels_filepath = root + 'TUSIMPLE/drivable_path.json'
    tusimple_images_filepath = root + 'TUSIMPLE/image/'

    # BDD100K - Data Loading
    bdd100k_Dataset = LoadDataEgoPath(bdd100k_labels_filepath, bdd100k_images_filepath, 'BDD100K')
    bdd100k_num_train_samples, bdd100k_num_val_samples = bdd100k_Dataset.getItemCount()
    bdd100k_sample_list = list(range(0, bdd100k_num_train_samples))
    random.shuffle(bdd100k_sample_list)

    # COMMA2K19 - Data Loading
    comma2k19_Dataset = LoadDataEgoPath(comma2k19_labels_fileapath, comma2k19_images_fileapath, 'COMMA2K19')
    comma2k19_num_train_samples, comma2k19_num_val_samples = comma2k19_Dataset.getItemCount()
    comma2k19_sample_list = list(range(0, comma2k19_num_train_samples))
    random.shuffle(comma2k19_sample_list)

    # CULANE - Data Loading
    culane_Dataset = LoadDataEgoPath(culane_labels_fileapath, culane_images_fileapath, 'CULANE')
    culane_num_train_samples, culane_num_val_samples = culane_Dataset.getItemCount()
    culane_sample_list = list(range(0, culane_num_train_samples))
    random.shuffle(culane_sample_list)

    # CURVELANES - Data Loading
    curvelanes_Dataset = LoadDataEgoPath(curvelanes_labels_fileapath, curvelanes_images_fileapath, 'CURVELANES')
    curvelanes_num_train_samples, curvelanes_num_val_samples = curvelanes_Dataset.getItemCount()
    curvelanes_sample_list = list(range(0, curvelanes_num_train_samples))
    random.shuffle(curvelanes_sample_list)

    # ROADWORK - Data Loading
    roadwork_Dataset = LoadDataEgoPath(roadwork_labels_fileapath, roadwork_images_fileapath, 'ROADWORK')
    roadwork_num_train_samples, roadwork_num_val_samples = roadwork_Dataset.getItemCount()
    roadwork_sample_list = list(range(0, roadwork_num_train_samples))
    random.shuffle(roadwork_sample_list)

    # TUSIMPLE - Data Loading
    tusimple_Dataset = LoadDataEgoPath(tusimple_labels_filepath, tusimple_images_filepath, 'TUSIMPLE')
    tusimple_num_train_samples, tusimple_num_val_samples = tusimple_Dataset.getItemCount()
    tusimple_sample_list = list(range(0, tusimple_num_train_samples))
    random.shuffle(tusimple_sample_list)

    # Total number of training samples
    total_train_samples = bdd100k_num_train_samples + \
    + comma2k19_num_train_samples + culane_num_train_samples \
    + curvelanes_num_train_samples + roadwork_num_train_samples \
    + tusimple_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = bdd100k_num_val_samples + \
    + comma2k19_num_val_samples + culane_num_val_samples \
    + curvelanes_num_val_samples + roadwork_num_val_samples \
    + tusimple_num_val_samples
    print(total_val_samples, ': total validation samples')

    # ====================== Training ====================== #

    # Trainer instance
    trainer = 0

    # Loading from checkpoint or training from scratch
    load_from_checkpoint= False
    pretrained_checkpoint_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/saves/SceneSeg/iter_140215_epoch_4_step_15999.pth'
    checkpoint_path = ''

    if(load_from_checkpoint == False):
        trainer = EgoPathTrainer(pretrained_checkpoint_path=pretrained_checkpoint_path)
    else:
        trainer = EgoPathTrainer(checkpoint_path=checkpoint_path, is_pretrained=False)

    # Zero gradients
    trainer.zero_grad()

    # Scale factors
    ENDPOINT_LOSS_SCALE_FACTOR = 1.0          
    MIDPOINT_LOSS_SCALE_FACTOR = 2.0
    GRADIENT_LOSS_SCALE_FACTOR = 0.5

    trainer.set_loss_scale_factors(ENDPOINT_LOSS_SCALE_FACTOR, 
            MIDPOINT_LOSS_SCALE_FACTOR, GRADIENT_LOSS_SCALE_FACTOR)
    
    trainer.set_gradient_loss_type('NUMERICAL')

  
    NUM_EPOCHS = 20
    LOGSTEP_LOSS = 50
    LOGSTEP_VIS = 50
    data_sampling_scheme = 'EQUAL'

    # Datasets list
    data_list = []
    data_list.append('BDD100K')
    data_list.append('COMMA2K19')
    data_list.append('CULANE')
    data_list.append('CURVELANES')
    data_list.append('ROADWORK')
    data_list.append('TUSIMPLE')
    
    batch_size = 24
    
    # Running thru epochs
    for epoch in range(0, NUM_EPOCHS):

        print('Epoch: ', epoch)

        # Batch Size Schedule
        if(epoch == 0):
            batch_size = 24
        elif(epoch == 1):
            batch_size = 12
        elif(epoch == 2):
            batch_size = 6
        elif(epoch == 3):
            batch_size = 3
        elif(epoch == 4):
            batch_size = 2
        elif(epoch > 4):
            batch_size = 1


        # Shuffle overall data list at start of epoch and reset data list counter
        random.shuffle(data_list)
        data_list_count = 0

        # Iterators for datasets
        bdd100k_count = 0
        comma2k19_count = 0
        culane_count = 0
        curvelanes_count = 0
        roadwork_count = 0
        tusimple_count = 0

        # Flags for whether all samples from a dataset have been read during the epoch
        is_bdd100k_complete = False
        is_comma2k19_complete = False
        is_culane_complete = False
        is_curvelanes_complete = False
        is_roadwork_complete = False
        is_tusimple_complete = False

        # Data sample counter
        count = 0

        # Loop through data
        while(True):

            # Log count
            count += 1
            log_count = count + total_train_samples*epoch

            # Reset iterators and shuffle individual datasets
            # based on data sampling scheme
            if(bdd100k_count == bdd100k_num_train_samples):
                
                if(data_sampling_scheme == 'EQUAL'):
                    bdd100k_count = 0
                    random.shuffle(bdd100k_sample_list)
                elif(data_sampling_scheme == 'CONCATENATE' and 
                        is_bdd100k_complete == False):
                    data_list.remove("BDD100K")

                is_bdd100k_complete = True
            
            if(comma2k19_count == comma2k19_num_train_samples):
                
                if(data_sampling_scheme == 'EQUAL'):
                    comma2k19_count = 0
                    random.shuffle(comma2k19_sample_list)
                elif(data_sampling_scheme == 'CONCATENATE' and 
                        is_comma2k19_complete == False):
                    data_list.remove('COMMA2K19')

                is_comma2k19_complete = True
            
            if(culane_count == culane_num_train_samples):
                
                if(data_sampling_scheme == 'EQUAL'):
                    culane_count = 0
                    random.shuffle(culane_sample_list)
                elif(data_sampling_scheme == 'CONCATENATE' and 
                        is_culane_complete == False):
                    data_list.remove('CULANE')    

                is_culane_complete = True

            if(curvelanes_count == curvelanes_num_train_samples):
                
                curvelanes_count = 0
                random.shuffle(curvelanes_sample_list)

                is_curvelanes_complete = True

            if(roadwork_count == roadwork_num_train_samples):
                
                if(data_sampling_scheme == 'EQUAL'):
                    roadwork_count = 0
                    random.shuffle(roadwork_sample_list)
                elif(data_sampling_scheme == 'CONCATENATE' and
                        is_roadwork_complete == False):
                    data_list.remove('ROADWORK')

                is_roadwork_complete = True

            if(tusimple_count == tusimple_num_train_samples):
                
                if(data_sampling_scheme == 'EQUAL'):
                    tusimple_count = 0
                    random.shuffle(tusimple_sample_list)
                elif(data_sampling_scheme == 'CONCATENATE' and
                        is_tusimple_complete == False):
                    data_list.remove('TUSIMPLE')

                is_tusimple_complete = True

            # If we have looped through each dataset at least once - restart the epoch
            if(is_bdd100k_complete and is_comma2k19_complete and is_culane_complete
               and is_curvelanes_complete and is_roadwork_complete and is_tusimple_complete):
                break
            
            # Reset the data list count if out of range
            if(data_list_count >= len(data_list)):
                data_list_count = 0

            # Get data depending on which dataset we are processing
            image = 0
            gt = 0
            is_valid = True

            if(data_list[data_list_count] == 'BDD100K'):
                image, gt, is_valid = bdd100k_Dataset.getItem(bdd100k_sample_list[bdd100k_count], is_train=True)
                bdd100k_count += 1

            if(data_list[data_list_count] == 'COMMA2K19'):
                image, gt, is_valid = comma2k19_Dataset.getItem(comma2k19_sample_list[comma2k19_count], is_train=True)
                comma2k19_count += 1

            if(data_list[data_list_count] == 'CULANE'):
                image, gt, is_valid = culane_Dataset.getItem(culane_sample_list[culane_count], is_train=True)
                culane_count += 1

            if(data_list[data_list_count] == 'CURVELANES'):
                image, gt, is_valid = curvelanes_Dataset.getItem(curvelanes_sample_list[curvelanes_count], is_train=True)
                curvelanes_count += 1

            if(data_list[data_list_count] == 'ROADWORK'):
                image, gt, is_valid = roadwork_Dataset.getItem(roadwork_sample_list[roadwork_count], is_train=True)
                roadwork_count += 1

            if(data_list[data_list_count] == 'TUSIMPLE'):
                image, gt, is_valid = tusimple_Dataset.getItem(tusimple_sample_list[tusimple_count], is_train=True)
                tusimple_count += 1
            
            if(is_valid):

                # Assign data
                trainer.set_data(image, gt)
                
                # Augment image
                trainer.apply_augmentations(is_train = True)
                
                # Converting to tensor and loading
                trainer.load_data()

                # Run model and get loss
                trainer.run_model()
                
                # Gradient accumulation
                trainer.loss_backward()

                # Simulating batch size through gradient accumulation
                if((count+1) % batch_size == 0):
                    trainer.run_optimizer()

                # Logging loss to Tensor Board
                if((count+1) % LOGSTEP_LOSS == 0):
                    trainer.log_loss(log_count)
                
                # Logging Visualization to Tensor Board
                if((count+1) % LOGSTEP_VIS == 0):  
                    trainer.save_visualization(log_count)
                '''
                # Save model and run val across entire val dataset
                if (current_index % LOGSTEP_MODEL == 0):

                    # Save model
                    model_save_path = os.path.join(
                        root_checkpoints,
                        f"iter_{log_index}_epoch_{epoch}_step_{i}.pth"
                    )
                    trainer.save_model(model_save_path)

                    # Validate
                    trainer.set_eval_mode()

                    # Metrics
                    running_total_loss = 0          # total_loss = mAE_gradients * alpha + mAE_endpoint
                    running_gradients_loss = 0      # MAE between gradients of GT and pred at uniform samples
                    running_endpoint_loss = 0       # MAE between GT start & end points with pred 
                    
                    # Temporarily disable gradient computation for backpropagation
                    with torch.no_grad():

                        # Compute val loss per dataset
                        for dataset, metadata in dict_data:

                            for val_index in range(metadata["N_vals"]):

                                # Fetch image and label for val set
                                image_val, label_val = metadata["loader_instance"].getItem(
                                    index = val_index,
                                    is_train = False
                                )

                                # Run val and calculate metrics
                                endpoint_loss, gradient_loss, total_loss = trainer.valudate(
                                    image_val,
                                    label_val
                                )

                                # Accumulate those metrics
                                running_endpoint_loss += endpoint_loss
                                running_gradients_loss += gradient_loss
                                running_total_loss += total_loss

                        # Compute average loss of complete val set
                        mAE_endpoint_loss = running_endpoint_loss / SUM_N_VALS
                        mAE_gradients_loss = running_gradients_loss / SUM_N_VALS
                        mAE_total_loss = running_total_loss / SUM_N_VALS

                        # Logging average metrics
                        trainer.log_mAE(
                            log_index,
                            mAE_endpoint_loss,
                            mAE_gradients_loss,
                            mAE_total_loss
                        )

                    # Switch back to training
                    trainer.set_train_mode()
                '''
                data_list_count += 1

    trainer.cleanup()
    

if __name__ == "__main__":
    main()

#%%