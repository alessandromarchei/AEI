#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import random
import pathlib
import sys
sys.path.append('..')
from data_utils.load_data_scene_3d import LoadDataScene3D
from training.scene_3d_trainer import Scene3DTrainer

def main():

    # Root path
    root = '/mnt/media/Scene3D/'

    # Model save path
    model_save_root_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/Scene3D/2025_02_28/model/'

    # Test images path
    test_images_save_root_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/Scene3D/test/'
    test_images_filepath = root + 'Test/'
    test_images = sorted([f for f in pathlib.Path(test_images_filepath).glob("*")])
    num_test_images = len(test_images)

    # Data path
    diverse_labels_filepath = root + 'Diverse/images-full-relative-depth/'
    diverse_images_filepath = root + 'Diverse/images-full/'

    # Data Loading
    Dataset = LoadDataScene3D(diverse_labels_filepath, diverse_images_filepath)
    total_train_samples, total_val_samples = Dataset.getItemCount()
    
    # Total train samples
    print(total_train_samples, ': Total training samples')

    # Total validation samples
    print(total_val_samples, ': Total validation samples')

    # Visual testing samples
    print(num_test_images, ': Total samples for visual testing')

    # Pre-trained model checkpoint path
    root_path = \
        '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/SceneSeg/run_1_batch_decay_Oct18_02-46-35/'
    pretrained_checkpoint_path = root_path + 'iter_140215_epoch_4_step_15999.pth'

    # Trainer Class
    trainer = Scene3DTrainer(pretrained_checkpoint_path=pretrained_checkpoint_path)
    trainer.zero_grad()
    
    # Total training epochs and batch size
    num_epochs = 20
    batch_size = 5

    # Epochs
    for epoch in range(0, num_epochs):

        print('Epoch: ', epoch + 1)
        randomlist_train_data = random.sample(range(0, total_train_samples), total_train_samples)

        # Learning Rate schedule            
        if(epoch >= 10):
            trainer.set_learning_rate(0.0000125)

        for count in range(0, total_train_samples):

            log_count = count + total_train_samples*epoch
            count += 1

            # Dataset sample 
            image, gt = Dataset.getItemTrain(randomlist_train_data[count])
            
            # Assign Data
            trainer.set_data(image, gt)
            
            # Augmenting Image
            trainer.apply_augmentations(is_train=True)

            # Converting to tensor and loading
            trainer.load_data()

            # Run model and calculate loss
            trainer.run_model()

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
            # dataset after 400000 steps
            if((log_count+1) % 500 == 0):
                
                # Save Model
                model_save_path = model_save_root_path + 'iter_' + \
                    str(log_count) \
                    + '_epoch_' +  str(epoch) + '_step_' + \
                    str(count) + '.pth'
                
                trainer.save_model(model_save_path)

                # Setting model to evaluation mode
                trainer.set_eval_mode()

                # Error
                mAE = 0
                mEL = 0

                # No gradient calculation
                with torch.no_grad():

                    # TEST
                    for test_count in range(0, num_test_images):
                        test_image = test_images[test_count]

                        test_save_path = test_images_save_root_path + 'iter_' + \
                            str(log_count) \
                            + '_epoch_' +  str(epoch) + '_step_' + \
                            str(count) + '_img_' + str(test_count) + '.png'
                        
                        trainer.test(test_image, test_save_path)

                    # VALIDATION
                    for val_count in range(0, total_val_samples):
                        image_val, gt_val = Dataset.getItemVal(val_count)

                        # Run Validation and calculate mAE Score
                        mAE_item, mEL_item = trainer.validate(image_val, gt_val)

                        # Accumulating mAE score
                        mAE += mAE_item
                        mEL += mEL_item

                    # LOGGING
                    # Calculating average loss of complete validation set for
                    # each specific dataset as well as the overall combined dataset
                    avg_mAE = mAE/total_val_samples
                    avg_mEL = mEL/total_val_samples
                    avg_overall = avg_mAE + avg_mEL
                    
                    # Logging average validation loss to TensorBoard
                    trainer.log_val_loss(avg_overall, avg_mAE, avg_mEL, log_count)

                # Resetting model back to training
                trainer.set_train_mode()
            
    trainer.cleanup()
    
    
if __name__ == '__main__':
    main()
# %%
