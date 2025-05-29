#!/usr/bin/env python3

import os
import torch
import random
from argparse import ArgumentParser
import sys
import torch.quantization

# Make sure we can find required modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from data_utils.load_data_scene_seg import LoadDataSceneSeg
from training.scene_seg_trainer import SceneSegTrainer
from quantize_components.scene_seg_network import SceneSegNetworkQuantized

def main_qat():
    parser = ArgumentParser(description="Quantization-Aware Training for Scene Segmentation")
    parser.add_argument("-s", "--model_save_root_path", dest="model_save_root_path", required=True, 
                        help="Root path where model checkpoints should be saved")
    parser.add_argument("-r", "--root", dest="root", required=True, 
                        help="Root path to folder where training data is stored")
    parser.add_argument("-fp32", "--fp32_model", dest="fp32_model_path", 
                        default="best_model.pth",
                        help="Path to pretrained FP32 model (default: best_model.pth in current dir)")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=5,
                        help="Number of QAT epochs (default: 5)")
    
    args = parser.parse_args()

    # QAT parameters with sensible defaults
    qat_params = {
        "learning_rate": 1e-5,
        "batch_size": 32,
        "backend": "fbgemm"  # x86 default
    }

    # Root path
    root = args.root

    # Model save path
    model_save_root_path = args.model_save_root_path

    # Data paths
    # ACDC
    acdc_labels_filepath = root + 'ACDC/ACDC/gt_masks/'
    acdc_images_filepath = root + 'ACDC/ACDC/images/'

    # IDDAW
    iddaw_labels_fileapath = root + 'IDDAW/IDDAW/gt_masks/'
    iddaw_images_fileapath = root + 'IDDAW/IDDAW/images/'

    # MUSES
    muses_labels_fileapath = root + 'MUSES/MUSES/gt_masks/'
    muses_images_fileapath = root + 'MUSES/MUSES/images/'

    # MAPILLARY
    mapillary_labels_fileapath = root + 'Mapillary_Vistas/Mapillary_Vistas/gt_masks/'
    mapillary_images_fileapath = root + 'Mapillary_Vistas/Mapillary_Vistas/images/'

    # COMMA10K
    comma10k_labels_fileapath = root + 'comma10k/comma10k/gt_masks/'
    comma10k_images_fileapath = root + 'comma10k/comma10k/images/'

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if qat_params["backend"] == "fbgemm" and device.type == "cuda":
        print("Note: Using fbgemm backend with CUDA. Model will be converted on CPU for final export.")

    # Initialize QAT Model
    print("Initializing QAT model...")
    model_qat = SceneSegNetworkQuantized().to(device)

    # Load Pretrained FP32 Weights
    fp32_path = args.fp32_model_path
    print(f"Loading pretrained weights from: {fp32_path}")
    try:
        fp32_checkpoint = torch.load(fp32_path, map_location=device)
        if isinstance(fp32_checkpoint, dict) and ("model_state_dict" in fp32_checkpoint or "state_dict" in fp32_checkpoint):
            fp32_state_dict = fp32_checkpoint.get("model_state_dict") or fp32_checkpoint.get("state_dict")
        else:
            fp32_state_dict = fp32_checkpoint
        model_qat.load_state_dict(fp32_state_dict, strict=False)
        print("Successfully loaded pretrained weights.")
    except Exception as e:
        print(f"Error loading FP32 checkpoint: {e}. Exiting.")
        return

    # Prepare Model for QAT
    print(f"Preparing model for QAT...")
    model_qat.qconfig = torch.quantization.get_default_qat_qconfig(qat_params["backend"])
    
    model_qat.train()
    print("Fusing model modules...")
    model_qat.fuse_model()
    
    print("Inserting fake quantization...")
    torch.quantization.prepare_qat(model_qat, inplace=True)
    print("Model prepared for QAT.")

    # ACDC - Data Loading
    acdc_Dataset = LoadDataSceneSeg(acdc_labels_filepath, acdc_images_filepath, 'ACDC')
    acdc_num_train_samples, acdc_num_val_samples = acdc_Dataset.getItemCount()

    # IDDAW - Data Loading
    iddaw_Dataset = LoadDataSceneSeg(iddaw_labels_fileapath, iddaw_images_fileapath, 'IDDAW')
    iddaw_num_train_samples, iddaw_num_val_samples = iddaw_Dataset.getItemCount()

    # MUSES - Data Loading
    muses_Dataset = LoadDataSceneSeg(muses_labels_fileapath, muses_images_fileapath, 'MUSES')
    muses_num_train_samples, muses_num_val_samples = muses_Dataset.getItemCount()

    # Mapillary - Data Loading
    mapillary_Dataset = LoadDataSceneSeg(mapillary_labels_fileapath, mapillary_images_fileapath, 'MAPILLARY')
    mapillary_num_train_samples, mapillary_num_val_samples = mapillary_Dataset.getItemCount()

    # comma10k - Data Loading
    comma10k_Dataset = LoadDataSceneSeg(comma10k_labels_fileapath, comma10k_images_fileapath, 'COMMA10K')
    comma10k_num_train_samples, comma10k_num_val_samples = comma10k_Dataset.getItemCount()

    # Total number of training samples
    total_train_samples = acdc_num_train_samples + \
    + iddaw_num_train_samples + muses_num_train_samples \
    + mapillary_num_train_samples + comma10k_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = acdc_num_val_samples + \
    + iddaw_num_val_samples + muses_num_val_samples \
    + mapillary_num_val_samples + comma10k_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Initialize Trainer
    print("Initializing trainer...")
    trainer = SceneSegTrainer() 
    trainer.model = model_qat
    trainer.device = device
    trainer.zero_grad()

    # Re-initialize optimizer with QAT learning rate
    print(f"Setting optimizer with LR: {qat_params['learning_rate']}")
    try:
        original_weight_decay = trainer.optimizer.defaults.get('weight_decay', 1e-5)
    except AttributeError:
        original_weight_decay = 1e-5

    trainer.optimizer = torch.optim.AdamW(
        trainer.model.parameters(), 
        lr=qat_params["learning_rate"], 
        weight_decay=original_weight_decay
    )
    
    # Total training epochs
    num_epochs = args.epochs
    batch_size = qat_params["batch_size"]

    # Epochs
    for epoch in range(0, num_epochs):
        print(f"QAT Epoch {epoch+1}/{num_epochs}")

        # Iterators for datasets
        acdc_count = 0
        iddaw_count = 0
        muses_count = 0
        comma10k_count = 0
        mapillary_count = 0

        is_acdc_complete = False
        is_iddaw_complete = False
        is_muses_complete = False
        is_mapillary_complete = False
        is_comma10k_complete = False

        data_list = []
        data_list.append('ACDC')
        data_list.append('IDDAW')
        data_list.append('MUSES')
        data_list.append('MAPILLARY')
        data_list.append('COMMA10K')
        random.shuffle(data_list)
        data_list_count = 0

        # Loop through data
        for count in range(0, total_train_samples):
            log_count = count + total_train_samples*epoch

            # Reset iterators
            if(acdc_count == acdc_num_train_samples and \
               is_acdc_complete == False):
                is_acdc_complete =  True
                if 'ACDC' in data_list:
                    data_list.remove("ACDC")
            
            if(iddaw_count == iddaw_num_train_samples and \
               is_iddaw_complete == False):
                is_iddaw_complete = True
                if 'IDDAW' in data_list:
                    data_list.remove("IDDAW")
            
            if(muses_count == muses_num_train_samples and \
                is_muses_complete == False):
                is_muses_complete = True
                if 'MUSES' in data_list:
                    data_list.remove('MUSES')
            
            if(mapillary_count == mapillary_num_train_samples and \
               is_mapillary_complete == False):
                is_mapillary_complete = True
                if 'MAPILLARY' in data_list:
                    data_list.remove('MAPILLARY')

            if(comma10k_count == comma10k_num_train_samples and \
               is_comma10k_complete == False):
                is_comma10k_complete = True
                if 'COMMA10K' in data_list:
                    data_list.remove('COMMA10K')

            if not data_list:
                print("All datasets exhausted for this epoch.")
                break

            if(data_list_count >= len(data_list)):
                data_list_count = 0

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            if(data_list[data_list_count] == 'ACDC' and \
                is_acdc_complete == False):
                image, gt, class_weights = \
                        acdc_Dataset.getItemTrain(acdc_count)
                acdc_count += 1
            
            elif(data_list[data_list_count] == 'IDDAW' and \
               is_iddaw_complete == False):
                image, gt, class_weights = \
                    iddaw_Dataset.getItemTrain(iddaw_count)      
                iddaw_count += 1

            elif(data_list[data_list_count] == 'MUSES' and \
               is_muses_complete == False):
                image, gt, class_weights = \
                    muses_Dataset.getItemTrain(muses_count)
                muses_count += 1
            
            elif(data_list[data_list_count] == 'MAPILLARY' and \
               is_mapillary_complete == False):
                image, gt, class_weights = \
                    mapillary_Dataset.getItemTrain(mapillary_count)
                mapillary_count +=1
            
            elif(data_list[data_list_count] == 'COMMA10K' and \
                is_comma10k_complete == False):
                image, gt, class_weights = \
                    comma10k_Dataset.getItemTrain(comma10k_count)
                comma10k_count += 1
            else:
                data_list_count += 1
                continue
            
            # Assign Data
            trainer.set_data(image, gt, class_weights)
            
            # Augmenting Image
            trainer.apply_augmentations(is_train=True)

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
            # dataset after 8000 steps
            if((count+1) % 8000 == 0 or (count+1) == total_train_samples):
                
                # Save Model
                model_save_path = model_save_root_path + '/qat_iter_' + \
                    str(count + total_train_samples*epoch) \
                    + '_epoch_' +  str(epoch) + '_step_' + \
                    str(count) + '.pth'
                
                torch.save(trainer.model.state_dict(), model_save_path)
                print(f"Saved checkpoint to {model_save_path}")
                
                # Validate
                print('Validating')

                # Setting model to evaluation mode
                trainer.set_eval_mode()

                running_IoU_full = 0
                running_IoU_bg = 0
                running_IoU_fg = 0
                running_IoU_rd = 0

                # No gradient calculation
                with torch.no_grad():

                    # ACDC
                    for val_count in range(0, acdc_num_val_samples):
                        image_val, gt_val, _ = \
                            acdc_Dataset.getItemVal(val_count)

                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # MUSES
                    for val_count in range(0, muses_num_val_samples):
                        image_val, gt_val, _ = \
                            muses_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd
                    
                    # IDDAW
                    for val_count in range(0, iddaw_num_val_samples):
                        image_val, gt_val, _ = \
                            iddaw_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # MAPILLARY
                    for val_count in range(0, mapillary_num_val_samples):
                        image_val, gt_val, _ = \
                            mapillary_Dataset.getItemVal(val_count)
                        
                         # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # COMMA10K
                    for val_count in range(0, comma10k_num_val_samples):
                        image_val, gt_val, _ = \
                            comma10k_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd
                    
                    # Calculating average loss of complete validation set
                    mIoU_full = running_IoU_full/total_val_samples
                    mIoU_bg = running_IoU_bg/total_val_samples
                    mIoU_fg = running_IoU_fg/total_val_samples
                    mIoU_rd = running_IoU_rd/total_val_samples
                    
                    # Logging average validation loss to TensorBoard
                    trainer.log_IoU(mIoU_full, mIoU_bg, mIoU_fg, mIoU_rd, log_count)

                # Resetting model back to training
                trainer.set_train_mode()
                
            data_list_count += 1

    # Convert to Quantized Model
    print("Converting to quantized model...")
    trainer.model.eval()
    cpu_model_qat = trainer.model.to('cpu')
    
    quantized_model = torch.quantization.convert(cpu_model_qat, inplace=False)
    print("Model conversion complete.")

    # Save final models
    final_model_path = os.path.join(
        model_save_root_path, 
        "quantized_model.pth"
    )
    torch.save(quantized_model.state_dict(), final_model_path)
    print(f"Saved quantized model to {final_model_path}")

    try:
        scripted_model_path = os.path.join(
            model_save_root_path, 
            "quantized_model_scripted.pt"
        )
        scripted_quantized_model = torch.jit.script(quantized_model)
        torch.jit.save(scripted_quantized_model, scripted_model_path)
        print(f"Saved scripted model to {scripted_model_path}")
    except Exception as e:
        print(f"Could not script model: {e}")

    if hasattr(trainer, 'cleanup'):
        trainer.cleanup()
    print("QAT training complete.")

if __name__ == '__main__':
    main_qat()
