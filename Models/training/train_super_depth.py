#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('..')
from data_utils.load_data_super_depth import LoadDataSuperDepth
from training.super_depth_trainer import SuperDepthTrainer


def main():

    # Root path
    root = '/mnt/media/SuperDepth/'

    # Model save path
    model_save_root_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SuperDepth/2025_01_14/'

    # Data paths
    # ARGOVERSE
    argoverse_labels_filepath = root + 'Argoverse/height/'
    argoverse_images_filepath = root + 'Argoverse/image/'
    argoverse_validities_filepath = root + 'Argoverse/validity/'

    # KITTI
    kitti_labels_filepath = root + 'KITTI/height/'
    kitti_images_filepath = root + 'KITTI/image/'
    kitti_validities_filepath = root + 'KITTI/validity/'

    # MUSES
    muses_labels_filepath = root + 'MUSES/height/'
    muses_images_filepath = root + 'MUSES/image/'
    muses_validities_filepath = root + 'MUSES/validity/'

    # DDAD
    ddad_labels_filepath = root + 'DDAD/height/'
    ddad_images_filepath = root + 'DDAD/image/'
    ddad_validities_filepath = root + 'DDAD/validity/'

    # URBANSYN
    urbansyn_labels_fileapath = root + 'UrbanSyn/height/'
    urbansyn_images_fileapath = root + 'UrbanSyn/image/'

    # ARGOVERSE - Data Loading
    argoverse_Dataset = LoadDataSuperDepth(argoverse_labels_filepath, argoverse_images_filepath, 
                                           'ARGOVERSE', argoverse_validities_filepath)
    argoverse_num_train_samples, argoverse_num_val_samples = argoverse_Dataset.getItemCount()

    # KITTI - Data Loading
    kitti_Dataset = LoadDataSuperDepth(kitti_labels_filepath, kitti_images_filepath, 
                                           'KITTI', kitti_validities_filepath)
    kitti_num_train_samples, kitti_num_val_samples = kitti_Dataset.getItemCount()

    # MUSES - Data Loading
    muses_Dataset = LoadDataSuperDepth(muses_labels_filepath, muses_images_filepath, 
                                           'MUSES', muses_validities_filepath)
    muses_num_train_samples, muses_num_val_samples = muses_Dataset.getItemCount()

    # DDAD - Data Loading
    ddad_Dataset = LoadDataSuperDepth(ddad_labels_filepath, ddad_images_filepath, 
                                           'DDAD', ddad_validities_filepath)
    ddad_num_train_samples, ddad_num_val_samples = ddad_Dataset.getItemCount()

    # URBANSYN - Data Loading
    urbansyn_Dataset = LoadDataSuperDepth(urbansyn_labels_fileapath, urbansyn_images_fileapath, 'URBANSYN')
    urbansyn_num_train_samples, urbansyn_num_val_samples = urbansyn_Dataset.getItemCount()

    # Total number of training samples
    total_train_samples = argoverse_num_train_samples + \
        kitti_num_train_samples + muses_num_train_samples + \
        ddad_num_train_samples + urbansyn_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = argoverse_num_val_samples + \
        kitti_num_val_samples + muses_num_val_samples + \
        ddad_num_val_samples + urbansyn_num_val_samples
    print(total_val_samples, ': total validation samples')

    
    # Pre-trained model checkpoint path
    root_path = \
        '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SceneSeg/run_1_batch_decay_Oct18_02-46-35/'
    pretrained_checkpoint_path = root_path + 'iter_140215_epoch_4_step_15999.pth'

    # Trainer Class
    trainer = SuperDepthTrainer(pretrained_checkpoint_path=pretrained_checkpoint_path)
    trainer.zero_grad()
    
    # Total training epochs
    num_epochs = 30
    batch_size = 5

    # Epochs
    for epoch in range(0, num_epochs):

        # Iterators for datasets
        argoverse_count = 0
        kitti_count = 0
        muses_count = 0
        ddad_count = 0
        urbansyn_count = 0

        is_argoverse_complete = False
        is_kitti_complete = False
        is_muses_complete = False
        is_ddad_complete = False
        is_urbansyn_complete = False
        
        data_list = []
        data_list.append('ARGOVERSE')
        data_list.append('KITTI')
        data_list.append('MUSES')
        data_list.append('DDAD')
        data_list.append('URBANSYN')
        random.shuffle(data_list)
        data_list_count = 0

        # Batch schedule
        if(epoch == 3):
            batch_size = 3
        
        if(epoch == 5):
            batch_size = 2
        
        if(epoch >= 10):
            batch_size = 1


        # Loop through data
        for count in range(0, total_train_samples):

            log_count = count + total_train_samples*epoch

            # Reset iterators
            if(argoverse_count == argoverse_num_train_samples and \
                is_argoverse_complete == False):
                is_argoverse_complete =  True
                data_list.remove("ARGOVERSE")

            if(kitti_count == kitti_num_train_samples and \
                is_kitti_complete == False):
                is_kitti_complete =  True
                data_list.remove("KITTI")

            if(muses_count == muses_num_train_samples and \
                is_muses_complete == False):
                is_muses_complete =  True
                data_list.remove("MUSES")
            
            if(ddad_count == ddad_num_train_samples and \
                is_ddad_complete == False):
                is_ddad_complete =  True
                data_list.remove("DDAD")
            
            if(urbansyn_count == urbansyn_num_train_samples and \
                is_urbansyn_complete == False):
                is_urbansyn_complete = True
                data_list.remove("URBANSYN")

            if(data_list_count >= len(data_list)):
                data_list_count = 0

            # Flag for whether dataset is simulated or not
            is_sim = False

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            if(data_list[data_list_count] == 'ARGOVERSE' and \
                is_argoverse_complete == False):
                image, gt, validity = argoverse_Dataset.getItemTrain(argoverse_count)
                argoverse_count += 1

            if(data_list[data_list_count] == 'KITTI' and \
                is_kitti_complete == False):
                image, gt, validity = kitti_Dataset.getItemTrain(kitti_count)
                kitti_count += 1

            if(data_list[data_list_count] == 'MUSES' and \
                is_muses_complete == False):
                image, gt, validity = muses_Dataset.getItemTrain(muses_count)
                muses_count += 1

            if(data_list[data_list_count] == 'DDAD' and \
                is_ddad_complete == False):
                image, gt, validity = ddad_Dataset.getItemTrain(ddad_count)
                ddad_count += 1
            
            if(data_list[data_list_count] == 'URBANSYN' and \
               is_urbansyn_complete == False):
                image, gt, validity = urbansyn_Dataset.getItemTrain(urbansyn_count)
                is_sim = True      
                urbansyn_count += 1

            # Assign Data
            trainer.set_data(image, gt, validity)
            
            # Augmenting Image
            trainer.apply_augmentations(is_train=True)

            # Converting to tensor and loading
            trainer.load_data(is_train=True)

            # Run model and calculate loss
            trainer.run_model(is_sim)

            # Gradient accumulation
            trainer.loss_backward()

            # Simulating batch size through gradient accumulation
            if((count+1) % batch_size == 0):
                trainer.run_optimizer()

            # Logging loss to Tensor Board every 250 steps
            if((count+1) % 250 == 0):
                trainer.log_loss(log_count)
            
            # Logging Image to Tensor Board every 1000 steps
            if((count+1) % 1000 == 0):  
                trainer.save_visualization(log_count)
            
            # Save model and run validation on entire validation 
            # dataset after 10000 steps
            if((count+1) % 1 == 0):
                
                # Save Model
                model_save_path = model_save_root_path + 'iter_' + \
                    str(count + total_train_samples*epoch) \
                    + '_epoch_' +  str(epoch) + '_step_' + \
                    str(count) + '.pth'
                
                trainer.save_model(model_save_path)
                
                # Validate
                print('Validating')

                # Setting model to evaluation mode
                trainer.set_eval_mode()

                # Error
                running_mAE_overall = 0
                running_mAE_argoverse = 0
                running_mAE_kitti = 0
                running_mAE_muses = 0
                running_mAE_ddad = 0
                running_mAE_urbansyn = 0
                
                # No gradient calculation
                with torch.no_grad():

                    # ARGOVERSE
                    #for val_count in range(0, argoverse_num_val_samples):
                    for val_count in range(0, 1):
                        image_val, gt_val, validity_val = argoverse_Dataset.getItemVal(val_count)

                        # Run Validation and calculate mAE Score
                        mAE = trainer.validate(image_val, gt_val, validity_val)

                        # Accumulating mAE score
                        running_mAE_argoverse += mAE
                        running_mAE_overall += mAE

                    # KITTI
                    #for val_count in range(0, kitti_num_val_samples):
                    for val_count in range(0, 1):
                        image_val, gt_val, validity_val = kitti_Dataset.getItemVal(val_count)

                        # Run Validation and calculate mAE Score
                        mAE = trainer.validate(image_val, gt_val, validity_val)

                        # Accumulating mAE score
                        running_mAE_kitti += mAE
                        running_mAE_overall += mAE

                    # MUSES
                    #for val_count in range(0, muses_num_val_samples):
                    for val_count in range(0, 1):
                        image_val, gt_val, validity_val = muses_Dataset.getItemVal(val_count)

                        # Run Validation and calculate mAE Score
                        mAE = trainer.validate(image_val, gt_val, validity_val)

                        # Accumulating mAE score
                        running_mAE_muses += mAE
                        running_mAE_overall += mAE

                    # DDAD
                    #for val_count in range(0, ddad_num_val_samples):
                    for val_count in range(0, 1):
                        image_val, gt_val, validity_val = ddad_Dataset.getItemVal(val_count)

                        # Run Validation and calculate mAE Score
                        mAE = trainer.validate(image_val, gt_val, validity_val)

                        # Accumulating mAE score
                        running_mAE_ddad += mAE
                        running_mAE_overall += mAE

                    # URBANSYN
                    #for val_count in range(0, urbansyn_num_val_samples):
                    for val_count in range(0, 1):
                        image_val, gt_val, validity_val = urbansyn_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate mAE Score
                        mAE = trainer.validate(image_val, gt_val, validity_val)

                        # Accumulating mAE score
                        running_mAE_urbansyn += mAE
                        running_mAE_overall += mAE

                    # LOGGING
                    # Calculating average loss of complete validation set for
                    # each specific dataset as well as the overall combined dataset
                    avg_mAE_overall = running_mAE_overall/total_val_samples
                    avg_mAE_argoverse = running_mAE_argoverse/argoverse_num_val_samples
                    avg_mAE_kitti = running_mAE_kitti/kitti_num_val_samples
                    avg_mAE_muses = running_mAE_muses/muses_num_val_samples
                    avg_mAE_ddad = running_mAE_ddad/ddad_num_val_samples
                    avg_mAE_urbansyn = running_mAE_urbansyn/urbansyn_num_val_samples

                    print('--- Validation Scores ---')
                    print('Overall: ', avg_mAE_overall)
                    print('ARGOVERSE: ', avg_mAE_argoverse)
                    print('KITTI: ', avg_mAE_kitti)
                    print('MUSES: ', avg_mAE_muses)
                    print('DDAD:', avg_mAE_ddad)
                    print('URBANSYN: ', avg_mAE_urbansyn)
                    
                    # Logging average validation loss to TensorBoard
                    trainer.log_val_mAE(avg_mAE_overall, avg_mAE_argoverse, avg_mAE_kitti, 
                        avg_mAE_muses, avg_mAE_ddad, avg_mAE_urbansyn, log_count)

                # Resetting model back to training
                trainer.set_train_mode()
            
            data_list_count += 1
            
    trainer.cleanup()
    
    
if __name__ == '__main__':
    main()
# %%
