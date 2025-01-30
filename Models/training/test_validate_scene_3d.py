#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import sys
sys.path.append('..')
from data_utils.load_data_scene_3d import LoadDataScene3D
from training.scene_3d_trainer import Scene3DTrainer

def main():

    # Root path
    root = '/mnt/media/Scene3D/'

    # Data paths

    # KITTI
    kitti_labels_filepath = root + 'KITTI/height/'
    kitti_images_filepath = root + 'KITTI/image/'
    kitti_validities_filepath = root + 'KITTI/validity/'

    # DDAD
    ddad_labels_filepath = root + 'DDAD/height/'
    ddad_images_filepath = root + 'DDAD/image/'
    ddad_validities_filepath = root + 'DDAD/validity/'

    # URBANSYN
    urbansyn_labels_fileapath = root + 'UrbanSyn/height/'
    urbansyn_images_fileapath = root + 'UrbanSyn/image/'

    # ARGOVERSE
    argoverse_labels_filepath = root + '/Argoverse/height/'
    argoverse_images_filepath = root + '/Argoverse/image/'

    # KITTI - Data Loading
    kitti_Dataset = LoadDataScene3D(kitti_labels_filepath, kitti_images_filepath, 
                                           'KITTI', kitti_validities_filepath)
    _, kitti_num_val_samples = kitti_Dataset.getItemCount()

    # DDAD - Data Loading
    ddad_Dataset = LoadDataScene3D(ddad_labels_filepath, ddad_images_filepath, 
                                           'DDAD', ddad_validities_filepath)
    _, ddad_num_val_samples = ddad_Dataset.getItemCount()

    # URBANSYN - Data Loading
    urbansyn_Dataset = LoadDataScene3D(urbansyn_labels_fileapath, urbansyn_images_fileapath, 'URBANSYN')
    _, urbansyn_num_val_samples = urbansyn_Dataset.getItemCount()

    # ARGOVERSE - Data Loading
    argoverse_Dataset = LoadDataScene3D(argoverse_labels_filepath, argoverse_images_filepath, 'ARGOVERSE')
    argoverse_num_train_samples, argoverse_num_val_samples = argoverse_Dataset.getItemCount()

    # Total validation samples
    total_val_samples = kitti_num_val_samples + \
        ddad_num_val_samples + urbansyn_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Total testing Samples
    total_test_samples = argoverse_num_train_samples + argoverse_num_val_samples
    print(total_test_samples, ': total testing samples')

    
    # Pre-trained model checkpoint path
    root_path = \
        '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/Scene3D/2025_01_27/model/'
    checkpoint_path = root_path + 'iter_1091507_epoch_26_step_39599.pth'

    
    # Trainer Class
    trainer = Scene3DTrainer(checkpoint_path=checkpoint_path, is_pretrained=True)
    trainer.zero_grad()
                
    # Validate
    print('Validating')

    # Setting model to evaluation mode
    trainer.set_eval_mode()

    # Error
    running_mAE_overall = 0
    running_mAE_kitti = 0
    running_mAE_ddad = 0
    running_mAE_urbansyn = 0
    running_mAE_argoverse = 0
    
    # No gradient calculation
    with torch.no_grad():

        # KITTI
        for val_count in range(0, kitti_num_val_samples):
            image_val, gt_val, validity_val = kitti_Dataset.getItemVal(val_count)

            # Run Validation and calculate mAE Score
            mAE = trainer.validate(image_val, gt_val, validity_val)

            # Accumulating mAE score
            running_mAE_kitti += mAE
            running_mAE_overall += mAE

        # DDAD
        for val_count in range(0, ddad_num_val_samples):
            image_val, gt_val, validity_val = ddad_Dataset.getItemVal(val_count)

            # Run Validation and calculate mAE Score
            mAE = trainer.validate(image_val, gt_val, validity_val)

            # Accumulating mAE score
            running_mAE_ddad += mAE
            running_mAE_overall += mAE

        # URBANSYN
        for val_count in range(0, urbansyn_num_val_samples):
            image_val, gt_val, validity_val = urbansyn_Dataset.getItemVal(val_count)
            
            # Run Validation and calculate mAE Score
            mAE = trainer.validate(image_val, gt_val, validity_val)

            # Accumulating mAE score
            running_mAE_urbansyn += mAE
            running_mAE_overall += mAE

        # Validate
        print('Testing')

        # ARGOVERSE
        for test_count in range(0, argoverse_num_val_samples):
            image_test, gt_test, validity_test = urbansyn_Dataset.getItemVal(test_count)
            
            # Run Validation and calculate mAE Score
            mAE = trainer.validate(image_test, gt_test, validity_test)

            # Accumulating mAE score
            running_mAE_argoverse += mAE

        for test_count in range(0, argoverse_num_train_samples):
            image_test, gt_test, validity_test = urbansyn_Dataset.getItemTrain(test_count)
            
            # Run Validation and calculate mAE Score
            mAE = trainer.validate(image_test, gt_test, validity_test)

            # Accumulating mAE score
            running_mAE_argoverse += mAE


        # LOGGING
        # Calculating average loss of complete validation set for
        # each specific dataset as well as the overall combined dataset
        avg_mAE_overall = running_mAE_overall/total_val_samples
        avg_mAE_kitti = running_mAE_kitti/kitti_num_val_samples
        avg_mAE_ddad = running_mAE_ddad/ddad_num_val_samples
        avg_mAE_urbansyn = running_mAE_urbansyn/urbansyn_num_val_samples
        avg_mAE_argoverse = running_mAE_argoverse/total_test_samples

        print('--- Validation Scores ---')
        print('Overall: ', avg_mAE_overall)
        print('KITTI: ', avg_mAE_kitti)
        print('DDAD:', avg_mAE_ddad)
        print('URBANSYN: ', avg_mAE_urbansyn)

        print('--- Testing Scores ---')
        print('ARGOVERSE: ', avg_mAE_argoverse)         
    
    
if __name__ == '__main__':
    main()
# %%
