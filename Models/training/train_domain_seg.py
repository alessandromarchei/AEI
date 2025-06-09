#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import random
from argparse import ArgumentParser
import sys
sys.path.append('..')
from data_utils.load_data_domain_seg import LoadDataDomainSeg
from training.domain_seg_trainer import DomainSegTrainer


def main():

    parser = ArgumentParser()
    parser.add_argument("-s", "--model_save_root_path", dest="model_save_root_path", help="root path where pytorch checkpoint file should be saved")
    parser.add_argument('-t', "--test_images_save_path", dest="test_images_save_path", help="path to where visualizations from inference on test images are saved")
    parser.add_argument("-r", "--root", dest="root", help="root path to folder where data training data is stored")
    args = parser.parse_args()

    # Root path
    root = args.root

    # Model save path
    model_save_root_path = args.model_save_root_path

    # Data paths
    # ROADWork data
    roadwork_labels_filepath = root + 'ROADWork/label/'
    roadwork_images_filepath = root + 'ROADWork/image/'

    # Test data
    test_images = root + '/Test/'
    test_images_save_path = args.test_images_save_path


    # ROADWork - Data Loading
    roadwork_Dataset = LoadDataDomainSeg(roadwork_labels_filepath, roadwork_images_filepath)
    roadwork_num_train_samples, roadwork_num_val_samples = roadwork_Dataset.getItemCount()

    # Total number of training samples
    total_train_samples = roadwork_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = roadwork_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Trainer Class
    trainer = DomainSegTrainer()
    trainer.zero_grad()
    
    # Total training epochs
    num_epochs = 20
    batch_size = 24

    # Epochs
    for epoch in range(0, num_epochs):

        # Printing epochs
        print('Epoch: ', epoch + 1)

        # Randomizing data
        randomlist_train_data = random.sample(range(0, total_train_samples), total_train_samples)

        # Batch size schedule
        if(epoch >= 1 and epoch < 5):
            batch_size = 16
        
        if(epoch >= 5 and epoch < 10):
            batch_size = 8
        
        if(epoch >= 10 and epoch < 15):
            batch_size = 4

        if (epoch >= 15 and epoch < 20):
            batch_size = 2

        if (epoch >= 20):
            batch_size = 1

        # Learning rate schedule
        if(epoch >= 10):
            trainer.set_learning_rate(0.000025)

        # Augmentations schedule
        apply_augmentations = True
        if(epoch >= 15):
            apply_augmentations = False

        # Loop through data
        for count in range(0, total_train_samples):

            # Log counter
            log_count = count + total_train_samples*epoch

    
            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            # Get data
            image, gt = roadwork_Dataset.getItemTrain(randomlist_train_data[count])
            
            # Assign Data
            trainer.set_data(image, gt)
            
            # Augmenting Image
            trainer.apply_augmentations(apply_augmentations)

            # Converting to tensor and loading
            trainer.load_data(is_train=True)

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
        # dataset after each epoch

        # Save Model
        model_save_path = model_save_root_path + 'iter_' + \
            str(count + total_train_samples*epoch) \
            + '_epoch_' +  str(epoch) + '_step_' + \
            str(count) + '.pth'
        
        trainer.save_model(model_save_path)

        # Test and save visualization
        print('Testing')
        trainer.test(test_images, test_images_save_path, log_count)
        
        # Validate
        print('Validating')

        # Setting model to evaluation mode
        trainer.set_eval_mode()

        # Overall IoU
        running_IoU = 0

        # No gradient calculation
        with torch.no_grad():

            # ACDC
            for val_count in range(0, roadwork_num_val_samples):
                image_val, gt_val = roadwork_num_val_samples.getItemVal(val_count)

                # Run Validation and calculate IoU Score
                IoU_score = trainer.validate(image_val, gt_val)

                # Accumulate individual IoU scores for validation samples
                running_IoU += IoU_score
            
            # Calculating average loss of complete validation set
            mIoU = running_IoU/total_val_samples
          
            # Logging average validation loss to TensorBoard
            trainer.log_IoU(mIoU, log_count)

        # Resetting model back to training
        trainer.set_train_mode()
            

    trainer.cleanup()
    

if __name__ == '__main__':
    main()
# %%
