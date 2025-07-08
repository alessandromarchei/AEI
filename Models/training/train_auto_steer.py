#! /usr/bin/env python3

import os
import torch
import random
import pathlib
from PIL import Image
from argparse import ArgumentParser
from typing import Literal, get_args
from matplotlib import pyplot as plt
import sys
sys.path.append('../..')
from Models.data_utils.load_data_auto_steer import LoadDataAutoSteer
from Models.training.auto_steer_trainer import AutoSteerTrainer

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
ORIG_VIS_PATH = "visualization"


def main():

    # ====================== Parsing input arguments ====================== #
    
    parser = ArgumentParser()

    parser.add_argument(
        "-r", "--root", 
        dest = "root", 
        required = True,
        help = "root path to folder where data training data is stored")
    
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
        required = False,
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
            "path_labels"   : os.path.join(ROOT_PATH, dataset, BEV_JSON_PATH),
            "path_images"   : os.path.join(ROOT_PATH, dataset, BEV_IMG_PATH),
            "path_orig_vis" : os.path.join(ROOT_PATH, dataset, ORIG_VIS_PATH)
        }

    # Deal with TEST dataset
    if (args.test_images_save_root_path):
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
        this_dataset_loader = LoadDataAutoSteer(
            labels_filepath = msdict[dataset]["path_labels"],
            images_filepath = msdict[dataset]["path_images"],
            dataset = dataset
        )
        N_trains, N_vals = this_dataset_loader.getItemCount()
        random_sample_list = random.sample(list(range(0, N_trains)), N_trains)

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

    
    CHECKPOINT_PATH = args.checkpoint_path

    if (CHECKPOINT_PATH):
        trainer = AutoSteerTrainer(checkpoint_path = CHECKPOINT_PATH)
    else:
      trainer = AutoSteerTrainer()
    
    # Zero gradients
    trainer.zero_grad()
    
    # Training loop parameters
    NUM_EPOCHS = 5
    LOGSTEP_LOSS = 250
    LOGSTEP_VIS = 1000
    LOGSTEP_MODEL = 10000

    # Val visualization param
    N_VALVIS = 50

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

    DATA_LOSS_BEV_SCALE_FACTOR = 1.0
    SMOOTHING_LOSS_BEV_SCALE_FACTOR = 10.0
    DATA_LOSS_SCALE_FACTOR = 1.0
    SMOOTHING_LOSS_SCALE_FACTOR = 10.0
    EGO_PATH_LOSS_SCALE_FACTOR = 1.0
    EGO_LANES_LOSS_SCALE_FACTOR = 1.0

    # Set training loss term scale factors
    trainer.set_loss_scale_factors(
        DATA_LOSS_BEV_SCALE_FACTOR,
        SMOOTHING_LOSS_BEV_SCALE_FACTOR,
        DATA_LOSS_SCALE_FACTOR,
        SMOOTHING_LOSS_SCALE_FACTOR,
        EGO_PATH_LOSS_SCALE_FACTOR,
        EGO_LANES_LOSS_SCALE_FACTOR

    )
    
    print(f"DATA_LOSS_SCALE_FACTOR : {DATA_LOSS_SCALE_FACTOR}")
    print(f"SMOOTHING_LOSS_SCALE_FACTOR : {SMOOTHING_LOSS_SCALE_FACTOR}")

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

    DATA_SAMPLING_SCHEME = "EQUAL" # EQUAL or CONCATENATE

    print(f"DATA_SAMPLING_SCHEME : {DATA_SAMPLING_SCHEME}")

    # BATCH_SIZE_SCHEME
    # There are three type of BATCH_SIZE_SCHEME, the 'CONSTANT' batch size
    # scheme sets a constant, fixed batch size value of 24 throughout training.
    # The 'SLOW_DECAY' batch size scheme reduces the batch size during training,
    # helping the model escape from local minima. The 'FAST_DECAY' batch size
    # scheme decays the batch size faster, and may help with quicker model
    # convergence.

    BATCH_SIZE_SCHEME = "FAST_DECAY" # FAST_DECAY or SLOW_DECAY or CONSTANT

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
            else:
                batch_size = 3
        elif (BATCH_SIZE_SCHEME == "SLOW_DECAY"):
            if (epoch == 0):
                batch_size = 24
            else:
                batch_size = 8

        else:
            raise ValueError(
                "Please speficy BATCH_SIZE_SCHEME as either " \
                " CONSTANT or FAST_DECAY or SLOW_DECAY"
            )
        
        # Learning Rate Schedule
        if ((epoch >= 1) and (epoch < 2)):
            trainer.set_learning_rate(0.000025)
        elif (epoch >= 2):
            trainer.set_learning_rate(0.0000125)
        # ((epoch >= 2) and (epoch < 3)):
        # elif (epoch >= 4):
        #     trainer.set_learning_rate(0.00000625)

        # Augmentation Schedule
        apply_augmentation = True
        if epoch >= 1:
            apply_augmentation = False

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
            msdict["sample_counter"] = msdict["sample_counter"] + 1
            msdict["log_counter"] = (
                msdict["sample_counter"] + \
                msdict["Nsum_trains"] * epoch
            )

            # Reset iterators and shuffle individual datasets
            # based on data sampling scheme
            for dataset in VALID_DATASET_LIST:
                N_trains = msdict[dataset]["N_trains"]
                if (msdict[dataset]["iter"] == N_trains):
                    if (DATA_SAMPLING_SCHEME == "EQUAL"):
                        msdict[dataset]["iter"] = 0

                        msdict[dataset]["sample_list"] = random.sample(list(range(0, N_trains)), N_trains)
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
            valids = []
            mat = []

            current_dataset = data_list[msdict["data_list_count"]]
            current_dataset_iter = msdict[current_dataset]["iter"]
            frame_id, image, xs, ys, _, valids, mat = msdict[current_dataset]["loader"].getItem(
                msdict[current_dataset]["sample_list"][current_dataset_iter],
                is_train = True
            )
            msdict[current_dataset]["iter"] = current_dataset_iter + 1

            # Also fetch original visualization
            orig_vis = Image.open(
                os.path.join(
                    msdict[dataset]["path_orig_vis"],
                    f"{frame_id}.jpg"
                )
            ).convert("RGB")

            # Start the training on this data

            # Assign data

            trainer.set_data(orig_vis, image, xs, ys, valids, mat)
            
            # Augment image
            trainer.apply_augmentations(apply_augmentation)
            
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
                trainer.log_loss(msdict["log_counter"] + 1)
            
            # Logging Visualization to Tensor Board
            if((msdict["sample_counter"] + 1) % LOGSTEP_VIS == 0):  
                trainer.save_visualization(msdict["log_counter"] + 1, orig_vis)
            
            # Save model and run Validation on entire validation dataset
            if ((msdict["sample_counter"] + 1) % LOGSTEP_MODEL == 0):
                
                print(f"\nIteration: {msdict['sample_counter'] + 1}")
                print("================ Saving Model ================")

                # Save model
                model_save_path = os.path.join(
                    MODEL_SAVE_ROOT_PATH,
                    f"iter_{msdict['log_counter'] + 1}_epoch_{epoch}_step_{msdict['sample_counter'] + 1}.pth"
                )
                trainer.save_model(model_save_path)
                
                # Set model to eval mode
                trainer.set_eval_mode()

                # Validation metrics for each dataset
                for dataset in VALID_DATASET_LIST:
                    msdict[dataset]["val_running"] = 0
                    msdict[dataset]["val_data_running"] = 0
                    msdict[dataset]["val_smooth_running"] = 0
                    msdict[dataset]["num_val_samples"] = 0

                # Temporarily disable gradient computation for backpropagation
                with torch.no_grad():

                    print("================ Running validation calculation ================")
                    
                    # Compute val loss per dataset
                    for dataset in VALID_DATASET_LIST:
                        for val_count in range(0, msdict[dataset]["N_vals"]):

                            # Fetch data
                            frame_id, image, xs, ys, _, valids, mat = msdict[dataset]["loader"].getItem(
                                val_count,
                                is_train = False
                            )
                            msdict[dataset]["num_val_samples"] = msdict[dataset]["num_val_samples"] + 1
                            
                            # Path handling
                            val_save_dir = os.path.join(
                                MODEL_SAVE_ROOT_PATH,
                                "VAL_VIS",
                                dataset,
                                f"iter_{msdict['log_counter'] + 1}_epoch_{epoch}_step_{msdict['sample_counter'] + 1}"
                            )
                            if not (os.path.exists(val_save_dir)):
                                os.makedirs(val_save_dir)

                            val_save_path = (
                                os.path.join(
                                    val_save_dir, 
                                    f"{str(val_count).zfill(2)}"
                                )
                                if (val_count < N_VALVIS)
                                else None
                            )

                            # Fetch it again, the orig vis
                            orig_vis = Image.open(
                                os.path.join(
                                    msdict[dataset]["path_orig_vis"],
                                    f"{frame_id}.jpg"
                                )
                            ).convert("RGB")

                            # Validate
                            val_metric, val_data, val_smooth = trainer.validate(
                                orig_vis, image, 
                                xs, ys, valids, mat,
                                val_save_path
                            )
                            
                            # Log
                            msdict[dataset]["val_running"] = msdict[dataset]["val_running"] + val_metric
                            msdict[dataset]["val_data_running"] = msdict[dataset]["val_data_running"] + val_data
                            msdict[dataset]["val_smooth_running"] = msdict[dataset]["val_smooth_running"] + val_smooth
                    
                    # Calculate final validation scores for network on each dataset
                    # as well as overall validation score - A lower score is better
                    for dataset in VALID_DATASET_LIST:
                        msdict[dataset]["val_score"] = msdict[dataset]["val_running"] / msdict[dataset]["num_val_samples"]
                        msdict[dataset]["val_data_score"] = msdict[dataset]["val_data_running"] / msdict[dataset]["num_val_samples"]
                        msdict[dataset]["val_smooth_score"] = msdict[dataset]["val_smooth_running"] / msdict[dataset]["num_val_samples"]

                    # Overall validation metric
                    msdict["val_overall_running"] = sum([
                        msdict[dataset]["val_running"]
                        for dataset in VALID_DATASET_LIST
                    ])
                    msdict["num_val_overall_samples"] = sum([
                        msdict[dataset]["num_val_samples"]
                        for dataset in VALID_DATASET_LIST
                    ])
                    msdict["overall_val_score"] = msdict["val_overall_running"] / msdict["num_val_overall_samples"]

                    # Overall validation data metric
                    msdict["val_overall_data_running"] = sum([
                        msdict[dataset]["val_data_running"]
                        for dataset in VALID_DATASET_LIST
                    ])
                    msdict["num_val_overall_samples"] = sum([
                        msdict[dataset]["num_val_samples"]
                        for dataset in VALID_DATASET_LIST
                    ])
                    msdict["overall_val_data_score"] = msdict["val_overall_data_running"] / msdict["num_val_overall_samples"]

                    msdict["val_overall_smooth_running"] = sum([
                        msdict[dataset]["val_smooth_running"]
                        for dataset in VALID_DATASET_LIST
                    ])
                    msdict["num_val_overall_samples"] = sum([
                        msdict[dataset]["num_val_samples"]
                        for dataset in VALID_DATASET_LIST
                    ])
                    msdict["overall_val_smooth_score"] = msdict[dataset]["val_smooth_running"] / msdict["num_val_overall_samples"]
                    
                    print("================ Complete - Validation Scores ================")
                    for dataset in VALID_DATASET_LIST:
                        print(f"\n{dataset} - VAL SCORE : {msdict[dataset]['val_score']}")
                        print(f"{dataset} - VAL DATA SCORE : {msdict[dataset]['val_data_score']}")
                        print(f"{dataset} - VAL SMOOTH SCORE : {msdict[dataset]['val_smooth_score']}")
                    print("\nOVERALL :")
                    print(f"VAL SCORE : {msdict['overall_val_score']}")
                    print(f"VAL DATA SCORE : {msdict['overall_val_data_score']}")
                    print(f"VAL SMOOTH SCORE : {msdict['overall_val_smooth_score']}\n")

                    # Logging average metrics
                    trainer.log_validation(msdict)

                # Switch back to training
                print("================ Continuing with training ================")
                trainer.set_train_mode()
            
            msdict["data_list_count"] = msdict["data_list_count"] + 1


if (__name__ == "__main__"):
    main()