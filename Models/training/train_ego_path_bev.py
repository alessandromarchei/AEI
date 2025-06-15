#! /usr/bin/env python3

import os
import torch
import random
import pathlib
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from typing import Literal, get_args
import sys
sys.path.append('../..')
from Models.data_utils.load_data_ego_path_bev import LoadDataBEVEgoPath
from Models.training.ego_path_trainer_bev import BEVEgoPathTrainer

# Currently limiting to available datasets only. Will unlock eventually
VALID_DATASET_LITERALS = Literal[
    # "BDD100K",
    # "COMMA2K19",
    # "CULANE",
    "CURVELANES",
    # "ROADWORK",
    # "TUSIMPLE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))

BEV_JSON_PATH = "drivable_path_bev.json"
BEV_IMG_PATH = "image_bev"


def main():

    # ====================== Parsing input arguments ====================== #
    
    parser = ArgumentParser()

    parser.add_argument(
        "-r", "--root", 
        dest = "root", 
        required = True,
        help = "root path to folder where data training data is stored")
    
    parser.add_argument(
        "-b", "--backbone_path", 
        dest = "backbone_path",
        help = "path to SceneSeg *.pth checkpoint file to load pre-trained backbone " \
        "if we are training EgoPath from scratch"
    )
    
    parser.add_argument(
        "-c", "--checkpoint_path", 
        dest = "checkpoint_path",
        help = "path to saved EgoPath *.pth checkpoint file for training from saved checkpoint"
    )

    parser.add_argument(
        "-s", "--model_save_root_path", 
        dest = "model_save_root_path",
        help = "root path where pytorch checkpoint file should be saved"
    )
    
    parser.add_argument(
        "-t", "--test_images_save_root_path", 
        dest = "test_images_save_root_path",
        help = "root path where test images should be saved"
    )

    args = parser.parse_args()

    # ====================== Loading datasets ====================== #

    # Root
    ROOT_PATH = args.root

    # Model save root path
    MODEL_SAVE_ROOT_PATH = args.model_save_root_path

    # Init metadata for datasets
    msdict = {}
    for dataset in VALID_DATASET_LIST:
        msdict[dataset] = {
            "path_labels" : os.path.join(ROOT_PATH, dataset, BEV_JSON_PATH),
            "path_images" : os.path.join(ROOT_PATH, dataset, BEV_IMG_PATH)
        }

    # Deal with TEST dataset
    msdict["TEST"] = {
        "list_images" : sorted([
            f for f in pathlib.Path(
                os.path.join(ROOT_PATH, "TEST")
            ).glob("*.png")
        ]),
        "path_test_save" : args.test_images_save_root_path
    }

    # Load datasets
    for dataset in VALID_DATASET_LIST:
        this_dataset_loader = LoadDataBEVEgoPath(
            labels_filepath = msdict[dataset]["path_labels"],
            images_filepath = msdict[dataset]["path_images"],
            dataset = dataset
        )
        N_trains, N_vals = this_dataset_loader.getItemCount()
        random_sample_list = random.shuffle(list(range(0, N_trains)))

        msdict[dataset]["loader"] = this_dataset_loader
        msdict[dataset]["N_trains"] = N_trains
        msdict[dataset]["N_vals"] = N_vals
        msdict[dataset]["sample_list"] = random_sample_list

        print(f"LOADED: {dataset} with {N_trains} train samples, {N_vals} val samples.")

    # All datasets - stats

    msdict["Nsum_trains"] = sum([
        msdict[dataset]["N_trains"]
        for dataset in VALID_DATASET_LIST
    ])
    print(f"Total train samples: {msdict['Nsum_trains']}")

    msdict["Nsum_vals"] = sum([
        msdict[dataset]["N_vals"]
        for dataset in VALID_DATASET_LIST
    ])
    print(f"Total val samples: {msdict['Nsum_vals']}")

    # ====================== Training params ====================== #

    # Trainer instance
    trainer = None

    BACKBONE_PATH = args.backbone_path
    CHECKPOINT_PATH = args.checkpoint_path

    if (BACKBONE_PATH and not CHECKPOINT_PATH):
        trainer = BEVEgoPathTrainer(pretrained_checkpoint_path = BACKBONE_PATH)
    elif (CHECKPOINT_PATH and not BACKBONE_PATH):
        trainer = BEVEgoPathTrainer(
            checkpoint_path = CHECKPOINT_PATH, 
            is_pretrained = True
        )
    elif (not CHECKPOINT_PATH and not BACKBONE_PATH):
        raise ValueError(
            "No checkpoint file found - Please ensure that you pass in either " \
            "a saved BEVEgoPath checkpoint file or SceneSeg checkpoint to load " \
            "the backbone weights"
        )
    else:
        raise ValueError(
            "Both BEVEgoPath checkpoint and SceneSeg checkpoint file provided - " \
            "please provide either the BEVEgoPath checkponit to continue training " \
            "from a saved checkpoint, or the SceneSeg checkpoint file to train " \
            "BEVEgoPath from scratch"
        )
    
    # Zero gradients
    trainer.zero_grad()
    
    # Training loop parameters
    NUM_EPOCHS = 20
    LOGSTEP_LOSS = 250
    LOGSTEP_VIS = 1000
    LOGSTEP_MODEL = 20000

    # MODIFIABLE PARAMETERS
    # You can adjust the SCALE FACTORS, GRAD_LOSS_TYPE, DATA_SAMPLING_SCHEME 
    # and BATCH_SIZE_DECAY during training

    # SCALE FACTORS
    # These scale factors impact the relative weight of different
    # loss terms in calculating the overall loss. A scale factor
    # value of 0.0 means that this loss is ignored, 1.0 means that
    # the loss is not scaled, and any other number applies a simple
    # scaling to increase or decrease the contribution of that specific
    # loss towards the overall loss

    DATA_LOSS_SCALE_FACTOR = 1.0
    SMOOTHING_LOSS_SCALE_FACTOR = 1.0
    FLAG_LOSS_SCALE_FACTOR = 1.0

    # Set training loss term scale factors
    trainer.set_loss_scale_factors(
        DATA_LOSS_SCALE_FACTOR,
        SMOOTHING_LOSS_SCALE_FACTOR,
        FLAG_LOSS_SCALE_FACTOR
    )
    
    print(f"DATA_LOSS_SCALE_FACTOR : {DATA_LOSS_SCALE_FACTOR}")
    print(f"SMOOTHING_LOSS_SCALE_FACTOR : {SMOOTHING_LOSS_SCALE_FACTOR}")
    print(f"FLAG_LOSS_SCALE_FACTOR : {FLAG_LOSS_SCALE_FACTOR}")

    # GRAD_LOSS_TYPE
    # There are two types of gradients loss, and either can be selected.
    # One option is 'NUMERICAL' which calculates the gradient through
    # the tangent angle between consecutive pairs of points along the
    # curve. The second option is 'ANALYTICAL' which uses the equation
    # of the curve to calculate the true mathematical gradient from
    # the curve's partial dervivatives

    GRAD_LOSS_TYPE = "NUMERICAL" # NUMERICAL or ANALYTICAL
    trainer.set_gradient_loss_type(GRAD_LOSS_TYPE)

    print(f"GRAD_LOSS_TYPE : {GRAD_LOSS_TYPE}")

    # DATA_SAMPLING_SCHEME
    # There are two data sampling schemes. The 'EQUAL' data sampling scheme
    # ensures that in each batch, we have an equal representation of samples
    # from each specific dataset. This schemes over-fits the network on 
    # smaller and underepresented datasets. The second sampling scheme is
    # 'CONCATENATE', in which the data is sampled randomly and the network
    # only sees each image from each dataset once in an epoch

    DATA_SAMPLING_SCHEME = "CONCATENATE" # EQUAL or CONCATENATE

    print(f"DATA_SAMPLING_SCHEME : {DATA_SAMPLING_SCHEME}")

    # BATCH_SIZE_SCHEME
    # There are three type of BATCH_SIZE_SCHEME, the 'CONSTANT' batch size
    # scheme sets a constant, fixed batch size value of 24 throughout training.
    # The 'SLOW_DECAY' batch size scheme reduces the batch size during training,
    # helping the model escape from local minima. The 'FAST_DECAY' batch size
    # scheme decays the batch size faster, and may help with quicker model
    # convergence.

    BATCH_SIZE_SCHEME = "SLOW_DECAY" # FAST_DECAY or SLOW_DECAY or CONSTANT

    print(f"BATCH_SIZE_SCHEME : {BATCH_SIZE_SCHEME}")
    
    # ======================================================================= #

    # ========================= Main training loop ========================= #

    # Batchsize
    batch_size = 0

    data_list = VALID_DATASET_LIST.copy()

    for epoch in range(0, NUM_EPOCHS):

        print(f"EPOCH : {epoch}")

        if (BATCH_SIZE_SCHEME == "CONSTANT"):
            batch_size = 3
        elif (BATCH_SIZE_SCHEME == "FAST_DECAY"):
            if (epoch == 0):
                batch_size = 24
            elif ((epoch >= 2) and (epoch < 4)):
                batch_size = 12
            elif ((epoch >= 4) and (epoch < 6)):
                batch_size = 6
            elif ((epoch >= 6) and (epoch < 8)):
                batch_size = 3
            elif ((epoch >= 8) and (epoch < 10)):
                batch_size = 2
            elif (epoch >= 10):
                batch_size = 1
        elif (BATCH_SIZE_SCHEME == "SLOW_DECAY"):
            if (epoch == 0):
                batch_size = 24
            elif ((epoch >= 2) and (epoch < 6)):
                batch_size = 12
            elif ((epoch >= 6) and (epoch < 10)):
                batch_size = 6
            elif ((epoch >= 10) and (epoch < 14)):
                batch_size = 3
            elif ((epoch >= 14) and (epoch < 18)):
                batch_size = 2
            elif (epoch >= 18):
                batch_size = 1
        else:
            raise ValueError(
                "Please speficy BATCH_SIZE_SCHEME as either " \
                " CONSTANT or FAST_DECAY or SLOW_DECAY"
            )
        
        # Learning Rate Schedule
        if ((epoch >= 6) and (epoch < 12)):
            trainer.set_learning_rate(0.00005)
        elif ((epoch >= 12) and (epoch < 18)):
            trainer.set_learning_rate(0.000025)
        elif (epoch >= 18):
            trainer.set_learning_rate(0.0000125)

        # Shuffle overall data list at start of epoch
        random.shuffle(data_list)
        msdict["data_list_count"] = 0
        
        # Reset all data counters
        msdict["sample_counter"] = 0
        for dataset in VALID_DATASET_LIST:
            msdict[dataset]["iter"] = 0
            msdict[dataset]["completed"] = False

        # Checking data sampling scheme
        if(
            (DATA_SAMPLING_SCHEME != "EQUAL") and 
            (DATA_SAMPLING_SCHEME != "CONCATENATE")
        ):
            raise ValueError(
                "Please speficy DATA_SAMPLING_SCHEME as either " \
                " EQUAL or CONCATENATE"
            )
        
        # Loop through data
        while (True):

            # Log count
            msdict["sample_counter"] += 1
            msdict["log_counter"] = (
                msdict["sample_counter"] + \
                msdict["Nsum_trains"] * epoch
            )

            # Reset iterators and shuffle individual datasets
            # based on data sampling scheme
            for dataset in VALID_DATASET_LIST:
                if (msdict[dataset]["iter"] == msdict[dataset]["N_trains"]):
                    if (DATA_SAMPLING_SCHEME == "EQUAL"):
                        msdict[dataset]["iter"] = 0
                        random.shuffle(msdict[dataset]["sample_list"])
                    elif (
                        (DATA_SAMPLING_SCHEME == "CONCATENATE") and 
                        (msdict[dataset]["completed"] == False)
                    ):
                        data_list.remove(dataset)

                    msdict[dataset]["completed"] = True

            # If we have looped through each dataset at least once - restart the epoch
            if (all([
                msdict[dataset]["completed"]
                for dataset in VALID_DATASET_LIST
            ])):
                break

            # Reset the data list count if out of range
            if (msdict["data_list_count"] >= len(data_list)):
                msdict["data_list_count"] = 0

            # Fetch data from current processed dataset
            
            image = None
            xs = []
            ys = []
            flags = None

            current_dataset = data_list[msdict["data_list_count"]]
            current_dataset_iter = msdict[current_dataset]["iter"]
            image, xs, ys, flags = msdict[current_dataset]["loader"].getItem(
                msdict[current_dataset]["sample_list"][current_dataset_iter],
                is_train = True
            )
            current_dataset_iter += 1

            # Start the training on this data

            # Assign data
            trainer.set_data(image, xs, ys, flags)
            
            # Augment image
            trainer.apply_augmentations(is_train = True)
            
            # Converting to tensor and loading
            trainer.load_data()

            # Run model and get loss
            trainer.run_model()
            
            # Gradient accumulation
            trainer.loss_backward()

            # Simulating batch size through gradient accumulation
            if ((msdict["sample_counter"] + 1) % batch_size == 0):
                trainer.run_optimizer()

            # Logging loss to Tensor Board
            if ((msdict["sample_counter"] + 1) % LOGSTEP_LOSS == 0):
                trainer.log_loss(log_count + 1)
            
            # Logging Visualization to Tensor Board
            if((count+1) % LOGSTEP_VIS == 0):  
                trainer.save_visualization(log_count+1)
            
            # Save model and run Validation on entire validation dataset
            if ((count+1) % LOGSTEP_MODEL == 0):
                
                print('\n')
                print('Iteration:', count+1)
                print('----- Saving Model -----')

                # Save model
                model_save_path = os.path.join(
                    model_save_root_path,
                    f"iter_{log_count+1}_epoch_{epoch}_step_{count+1}.pth"
                )
                trainer.save_model(model_save_path)
                
                # Set model to eval mode
                trainer.set_eval_mode()

                # # Running test
                # print('----- Running Testing -----')
                # for i in range(0, len(test_images_list)):
                    
                #     test_image_save_path = os.path.join(
                #         test_images_save_root_path,
                #         f"iter_{log_count+1}_epoch_{epoch}_step_{count+1}_{i}.png"
                #     )

                #     test_image_path = str(test_images_list[i])
                #     trainer.test(test_image_path, test_image_save_path)

                # Validation metrics for each dataset
                # val_bdd100k_running = 0
                # num_val_bdd100k_samples = 0

                # val_comma2k19_running = 0
                # num_val_comma2k19_samples = 0

                # val_culane_running = 0
                # num_val_culane_samples = 0

                val_curvelanes_running = 0
                num_val_curvelanes_samples = 0

                # val_roadwork_running = 0
                # num_val_roadwork_samples = 0

                # val_tusimple_running = 0
                # num_val_tusimple_samples = 0

                # Temporarily disable gradient computation for backpropagation
                with torch.no_grad():

                    print('----- Running validation calculation -----')
                    # Compute val loss per dataset
                    
                    # # BDD100K
                    # for val_count in range(0, bdd100k_num_val_samples):
                    #     image, gt, is_valid = bdd100k_Dataset.getItem(val_count, is_train=False)

                    #     if(is_valid):
                    #         num_val_bdd100k_samples += 1
                    #         val_metric = trainer.validate(image, gt)
                    #         val_bdd100k_running = val_bdd100k_running + val_metric
                    
                    # # COMMA2K19
                    # for val_count in range(0, comma2k19_num_val_samples):
                    #     image, gt, is_valid = comma2k19_Dataset.getItem(val_count, is_train=False)

                    #     if(is_valid):
                    #         num_val_comma2k19_samples += 1
                    #         val_metric = trainer.validate(image, gt)
                    #         val_comma2k19_running = val_comma2k19_running + val_metric
                    
                    # # CULANE
                    # for val_count in range(0, culane_num_val_samples):
                    #     image, gt, is_valid = culane_Dataset.getItem(val_count, is_train=False)

                    #     if(is_valid):
                    #         num_val_culane_samples += 1
                    #         val_metric = trainer.validate(image, gt)
                    #         val_culane_running = val_culane_running + val_metric
                    
                    # CURVELANES
                    for val_count in range(0, curvelanes_num_val_samples):
                        image, gt, is_valid = curvelanes_Dataset.getItem(val_count, is_train=False)

                        if(is_valid):
                            num_val_curvelanes_samples += 1
                            val_metric = trainer.validate(image, gt)
                            val_curvelanes_running = val_curvelanes_running + val_metric
                    
                    # # ROADWORK
                    # for val_count in range(0, roadwork_num_val_samples):
                    #     image, gt, is_valid = roadwork_Dataset.getItem(val_count, is_train=False)

                    #     if(is_valid):
                    #         num_val_roadwork_samples += 1
                    #         val_metric = trainer.validate(image, gt)
                    #         val_roadwork_running = val_roadwork_running + val_metric
                    
                    # # TUSIMPLE
                    # for val_count in range(0, tusimple_num_val_samples):
                    #     image, gt, is_valid = tusimple_Dataset.getItem(val_count, is_train=False)

                    #     if(is_valid):
                    #         num_val_tusimple_samples += 1
                    #         val_metric = trainer.validate(image, gt)
                    #         val_tusimple_running = val_tusimple_running + val_metric
                    
                    # Calculate final validation scores for network on each dataset
                    # as well as overall validation score - A lower score is better
                    # bdd100k_val_score = val_bdd100k_running/num_val_bdd100k_samples
                    # comma2k19_val_score = val_comma2k19_running/num_val_comma2k19_samples
                    # culane_val_score = val_culane_running/num_val_culane_samples
                    curvelanes_val_score = val_curvelanes_running/num_val_curvelanes_samples
                    # roadwork_val_score = val_roadwork_running/num_val_roadwork_samples
                    # tusimple_val_score = val_tusimple_running/num_val_tusimple_samples

                    # Ovearll validation metric
                    # val_overall_running = val_bdd100k_running + val_comma2k19_running + \
                    #     val_culane_running + val_curvelanes_running + val_roadwork_running + \
                    #     val_tusimple_running
                    val_overall_running = val_curvelanes_running
                    
                    # num_val_overall_samples = num_val_bdd100k_samples + num_val_comma2k19_samples + \
                    #     num_val_culane_samples + num_val_curvelanes_samples + num_val_roadwork_samples + \
                    #     num_val_tusimple_samples
                    num_val_overall_samples = num_val_curvelanes_samples
                    
                    overall_validation_score = val_overall_running/num_val_overall_samples
                    
                    print('---------- Complete - Validation Scores ----------')
                    # print('BDD100K: ', bdd100k_val_score)
                    # print('COMM2K19: ', comma2k19_val_score)
                    # print('CULANE: ', culane_val_score)
                    # print('CURVELANES: ', curvelanes_val_score)
                    # print('ROADWORK: ', roadwork_val_score)
                    # print('TUSIMPLE: ', tusimple_val_score)
                    print('OVERALL:', overall_validation_score)
                    print('\n')

                    # Logging average metrics
                    trainer.log_validation(
                        log_count, 
                        # bdd100k_val_score, 
                        # comma2k19_val_score,
                        # culane_val_score, 
                        # curvelanes_val_score, 
                        # roadwork_val_score,
                        # tusimple_val_score, 
                        overall_validation_score
                    )

                # Switch back to training
                print('----- Continuing with training -----')
                trainer.set_train_mode()
            
            data_list_count += 1