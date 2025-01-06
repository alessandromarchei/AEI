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

    # MUAD - Data Loading
    muad_Dataset = LoadDataSuperDepth(muad_labels_filepath, muad_images_filepath, 'MUAD')
    _, muad_num_val_samples = muad_Dataset.getItemCount()

    # BDD100K - Data Loading
    urbansyn_Dataset = LoadDataSuperDepth(urbansyn_labels_fileapath, urbansyn_images_fileapath, 'URBANSYN')
    _, urbansyn_num_val_samples = urbansyn_Dataset.getItemCount()

    # Total number of validation samples
    total_val_samples = muad_num_val_samples + urbansyn_num_val_samples
    print(total_val_samples, ': total validation samples')

    # KITTI
    kitti_gt_files = sorted([f for f in pathlib.Path(root + 'KITTI/height_fill/').glob("*.npy")])
    kitti_img_files = sorted([f for f in pathlib.Path(root + 'KITTI/image/').glob("*.png")])
    kitti_valid_files = sorted([f for f in pathlib.Path(root + 'KITTI/height_validity').glob("*.png")])

    # Total number of test samples
    num_test_gt_files = len(kitti_gt_files)
    num_test_img_files = len(kitti_img_files)
    num_test_valid_files = len(kitti_valid_files)

    checkDataImg = CheckData(num_test_gt_files, num_test_img_files)
    checkDataValid = CheckData(num_test_gt_files, num_test_valid_files)

    if(checkDataImg.getCheck() and checkDataValid.getCheck):
        print(num_test_gt_files, ': total test samples')

        # Loading models
        pretrained_model_root_path = \
            '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SceneSeg/run_1_batch_decay_Oct18_02-46-35/'
        pretrained_checkpoint_path = pretrained_model_root_path + 'iter_140215_epoch_4_step_15999.pth'

        model_root_path = \
            '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SuperDepth/'
        checkpoint_path = model_root_path + 'iter_118959_epoch_12_step_3999.pth'
        
        # Trainer Class
        trainer = SuperDepthTrainer(checkpoint_path=checkpoint_path, 
            pretrained_checkpoint_path=pretrained_checkpoint_path)
        trainer.zero_grad()

        # Setting model to evaluation mode
        trainer.set_eval_mode()

        # VALIDATION

        # Error
        muad_running_mAE = 0
        urbansyn_running_mAE = 0
        overall_running_mAE = 0

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

            # LOGGING
            # Calculating average loss of complete validation set
            avg_muad_mAE = muad_running_mAE/muad_num_val_samples
            print('MUAD average validation error:', avg_muad_mAE)

            avg_urbansyn_mAE = urbansyn_running_mAE/urbansyn_num_val_samples
            print('Urbansyn average validation error:', avg_urbansyn_mAE)

            avg_overall_mAE = overall_running_mAE/total_val_samples
            print('Overall average validation error:', avg_overall_mAE)


if __name__ == '__main__':
    main()
# %%
