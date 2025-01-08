#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import pathlib
import sys
sys.path.append('..')
from data_utils.load_data_super_depth import LoadDataSuperDepth
from data_utils.check_data import CheckData
from training.super_depth_trainer import SuperDepthTrainer

def main():

    # Root path
    root = '/mnt/media/SuperDepth/'
    
    # Data paths
    # MUAD
    muad_labels_filepath = root + 'MUAD/height/'
    muad_images_filepath = root + 'MUAD/image/'

    # URBANSYN
    urbansyn_labels_fileapath = root + 'UrbanSyn/height/'
    urbansyn_images_fileapath = root + 'UrbanSyn/image/'

    # KITTI
    kitti_labels_filepath = root + 'KITTI/height_fill/'
    kitti_images_filepath = root + 'KITTI/image/'
    kitti_validity_filepath = root + 'KITTI/height_validity/'

    # MUAD - Data Loading
    muad_Dataset = LoadDataSuperDepth(muad_labels_filepath, muad_images_filepath, 'MUAD')
    _, muad_num_val_samples = muad_Dataset.getItemCount()

    # BDD100K - Data Loading
    urbansyn_Dataset = LoadDataSuperDepth(urbansyn_labels_fileapath, urbansyn_images_fileapath, 'URBANSYN')
    _, urbansyn_num_val_samples = urbansyn_Dataset.getItemCount()

    # Total number of validation samples
    total_val_samples = muad_num_val_samples + urbansyn_num_val_samples
    print(total_val_samples, ': total validation samples')

    # KITTI - Data Loading
    kitti_Dataset = LoadDataSuperDepth(kitti_labels_filepath, kitti_images_filepath, 'KITTI_TEST', kitti_validity_filepath)
    total_test_samples = kitti_Dataset.getTestCount()

    # Total number of test samples
    print(total_test_samples, ': total test samples')

    # Loading model
    model_root_path = \
        '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SuperDepth/'
    checkpoint_path = model_root_path + 'iter_118959_epoch_12_step_3999.pth'
    
    # Trainer Class
    trainer = SuperDepthTrainer(checkpoint_path=checkpoint_path, is_pretrained = True)
    
    trainer.zero_grad()

    # Setting model to evaluation mode
    trainer.set_eval_mode()

    # VALIDATION

    # Error
    muad_running_mAE = 0
    urbansyn_running_mAE = 0
    overall_running_mAE = 0
    test_running_mAE = 0

    # No gradient calculation
    with torch.no_grad():

        # MUAD
        for val_count in range(0, muad_num_val_samples):
            image_val, gt_val = muad_Dataset.getItemVal(val_count)

            # Run Validation and calculate mAE Score
            mAE = trainer.validate(image_val, gt_val)

            # Accumulating mAE score
            muad_running_mAE += mAE
            overall_running_mAE += mAE


        # URBANSYN
        for val_count in range(0, urbansyn_num_val_samples):
            image_val, gt_val = urbansyn_Dataset.getItemVal(val_count)
            
            # Run Validation and calculate mAE Score
            mAE = trainer.validate(image_val, gt_val)

            # Accumulating mAE score
            urbansyn_running_mAE += mAE
            overall_running_mAE += mAE

        # KITTI-TEST
        for test_count in range(0, total_test_samples):
            image_test, gt_test, validity_test = kitti_Dataset.getItemTest(test_count)
            
            # Run Validation and calculate mAE Score
            mAE = trainer.test(image_test, gt_test, validity_test)

            # Accumulating mAE score
            test_running_mAE += mAE    

        # LOGGING
        # Calculating average loss of complete validation set
        print('Model: ', checkpoint_path)

        avg_muad_mAE = muad_running_mAE/muad_num_val_samples
        print('MUAD average validation error:', avg_muad_mAE)

        avg_urbansyn_mAE = urbansyn_running_mAE/urbansyn_num_val_samples
        print('Urbansyn average validation error:', avg_urbansyn_mAE)

        avg_overall_mAE = overall_running_mAE/total_val_samples
        print('Overall average validation error:', avg_overall_mAE)

        avg_test_mAE = test_running_mAE/total_test_samples
        print('Average test validation error:', avg_test_mAE)


if __name__ == '__main__':
    main()
# %%
