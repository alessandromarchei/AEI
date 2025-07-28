#! /usr/bin/env python3

import os
import torch
import random
import pathlib
from PIL import Image
#from argparse import ArgumentParser
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
    # "CURVELANES",
    # "ROADWORK",
    "TUSIMPLE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))

BEV_JSON_PATH = "drivable_path_bev.json"
BEV_IMG_PATH = "image_bev"
ORIG_VIS_PATH = "visualization"


def main():

    # ====================== Parsing input arguments ====================== #
    '''
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
        required = False,
        dest = "test_images_save_root_path",
        help = "root path where test images should be saved"
    )

    args = parser.parse_args()
    '''
    # ====================== Loading datasets ====================== #

    # Root
    ROOT_PATH = '/home/zain/Autoware/Data/AutoSteer/'#args.root

    # Model save root path
    MODEL_SAVE_ROOT_PATH = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/saves/AutoSteer/' args.model_save_root_path

    # Init metadata for datasets
    msdict = {}
    for dataset in VALID_DATASET_LIST:
        msdict[dataset] = {
            "path_labels"   : os.path.join(ROOT_PATH, dataset, BEV_JSON_PATH),
            "path_images"   : os.path.join(ROOT_PATH, dataset, BEV_IMG_PATH),
            "path_orig_vis" : os.path.join(ROOT_PATH, dataset, ORIG_VIS_PATH)
        }

    # Deal with TEST dataset
    #if (args.test_images_save_root_path):
    #    msdict["TEST"] = {
    #        "list_images" : sorted([
    #            f for f in pathlib.Path(
    #                os.path.join(ROOT_PATH, "TEST")
    #            ).glob("*.png")
    #        ]),
    #        "path_test_save" : args.test_images_save_root_path
    #    }

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

    CHECKPOINT_PATH = None #args.checkpoint_path

    if (CHECKPOINT_PATH):
        trainer = AutoSteerTrainer(checkpoint_path = CHECKPOINT_PATH)    
    else:
        trainer = AutoSteerTrainer()
    
    # Zero gradients
    trainer.zero_grad()
    
    # Training loop parameters
    NUM_EPOCHS = 1
    LOGSTEP_LOSS = 250
    LOGSTEP_VIS = 1000
    LOGSTEP_MODEL = 5000

    # Val visualization param
    N_VALVIS = 50


    # SCALE FACTORS
    # These scale factors impact the relative weight of different
    # loss terms in calculating the overall loss. A scale factor
    # value of 0.0 means that this loss is ignored, 1.0 means that
    # the loss is not scaled, and any other number applies a simple
    # scaling to increase or decrease the contribution of that specific
    # loss towards the overall loss
    
    BEV_GRADIENT_SCALE = 1.0
    PERSPECTIVE_GRADIENT_SCALE = 1.0
    OVERALL_SCALE = 1.0


    # Set training loss term scale factors
    trainer.set_loss_scale_factors(
        BEV_GRADIENT_SCALE,
        PERSPECTIVE_GRADIENT_SCALE,
        OVERALL_SCALE
    )
    
    print(f"BEV_GRADIENT_SCALE : {BEV_GRADIENT_SCALE}")
    print(f"PERSPECTIVE_GRADIENT_SCALE : {PERSPECTIVE_GRADIENT_SCALE}")
    print(f"OVERALL_SCALE : {OVERALL_SCALE}")

    
    # ========================= Main training loop ========================= #

    # Batch Size
    batch_size = 32

    data_list = VALID_DATASET_LIST.copy()

    for epoch in range(0, NUM_EPOCHS):

        print(f"EPOCH : {epoch}")

        # Batch Size Schedule
        if (epoch > 10 and epoch <= 20):
            batch_size = 16
        elif (epoch > 20 and epoch <= 30):
            batch_size = 8
        elif (epoch > 30):
            batch_size = 4
      
        # Learning Rate Schedule
        if(epoch < 30):
            trainer.set_learning_rate(0.0005)
        elif (epoch >= 30 and epoch < 40):
            trainer.set_learning_rate(0.0001)
        elif (epoch >= 40):
            trainer.set_learning_rate(0.000025)

        # Augmentation Schedule
        apply_augmentation = False
        if ((epoch >= 15) and (epoch < 35)):
            apply_augmentation = True

        # Shuffle overall data list at start of epoch
        random.shuffle(data_list)
        msdict["data_list_count"] = 0
        
        # Reset all data counters
        msdict["sample_counter"] = 0
        for dataset in VALID_DATASET_LIST:
            msdict[dataset]["iter"] = 0
            msdict[dataset]["completed"] = False

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
                    if ((msdict[dataset]["completed"] == False)):
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
            xs_bev_egopath = []
            xs_reproj_egopath = []
            xs_bev_egoleft = []
            xs_reproj_egoleft = []
            xs_bev_egoright = []
            xs_reproj_egoright = []
            ys_bev = []
            ys_reproj = []
            flags_egopath = []
            valids_egopath = []
            flags_egoleft = []
            valids_egoleft = []
            flags_egoright = []
            valids_egoright = []

            current_dataset = data_list[msdict["data_list_count"]]
            current_dataset_iter = msdict[current_dataset]["iter"]
            [
                frame_id, image,
                xs_bev_egopath, xs_reproj_egopath,
                xs_bev_egoleft, xs_reproj_egoleft,
                xs_bev_egoright, xs_reproj_egoright,
                ys_bev, ys_reproj,
                flags_egopath, valids_egopath,
                flags_egoleft, valids_egoleft,
                flags_egoright, valids_egoright,
                transform_matrix
            ] = msdict[current_dataset]["loader"].getItem(
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

            trainer.set_data(
                orig_vis, image, 
                xs_bev_egopath,
                xs_reproj_egopath,
                xs_bev_egoleft,
                xs_reproj_egoleft,
                xs_bev_egoright,
                xs_reproj_egoright,
                ys_bev,
                ys_reproj,
                valids_egopath,
                valids_egoleft,
                valids_egoright,
                transform_matrix
            )
            
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
                    val_dict_schema = {
                        "total_running" : 0,
                        "total_score" : 0,
                        "bev_running" : 0,
                        "bev_score" : 0,
                        "reproj_running" : 0,
                        "reproj_score" : 0
                    }
                    msdict[dataset]["val_egopath"] = val_dict_schema
                    msdict[dataset]["val_egoleft"] = val_dict_schema
                    msdict[dataset]["val_egoright"] = val_dict_schema
                    msdict[dataset]["num_val_samples"] = 0

                # Temporarily disable gradient computation for backpropagation
                with torch.no_grad():

                    print("================ Running validation calculation ================")
                    
                    # Compute val loss per dataset
                    for dataset in VALID_DATASET_LIST:
                        for val_count in range(0, msdict[dataset]["N_vals"]):

                            # Fetch data
                            [
                                frame_id, image,
                                xs_bev_egopath, xs_reproj_egopath,
                                xs_bev_egoleft, xs_reproj_egoleft,
                                xs_bev_egoright, xs_reproj_egoright,
                                ys_bev, ys_reproj,
                                flags_egopath, valids_egopath,
                                flags_egoleft, valids_egoleft,
                                flags_egoright, valids_egoright,
                                transform_matrix
                            ] = msdict[dataset]["loader"].getItem(
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
                            [
                                val_total_loss_egopath, val_bev_loss_egopath, val_reproj_loss_egopath,
                                val_total_loss_egoleft, val_bev_loss_egoleft, val_reproj_loss_egoleft,
                                val_total_loss_egoright, val_bev_loss_egoright, val_reproj_loss_egoright
                            ] = trainer.validate(
                                orig_vis, image, 
                                xs_bev_egopath,
                                xs_reproj_egopath,
                                xs_bev_egoleft,
                                xs_reproj_egoleft,
                                xs_bev_egoright,
                                xs_reproj_egoright,
                                ys_bev,
                                ys_reproj,
                                valids_egopath,
                                valids_egoleft,
                                valids_egoright,
                                transform_matrix,
                                val_save_path
                            )
                            
                            # Log

                            # Egopath
                            msdict[dataset]["val_egopath"]["total_running"] += val_total_loss_egopath
                            msdict[dataset]["val_egopath"]["bev_running"] += val_bev_loss_egopath
                            msdict[dataset]["val_egopath"]["reproj_running"] += val_reproj_loss_egopath

                            # Egoleft
                            msdict[dataset]["val_egoleft"]["total_running"] += val_total_loss_egoleft
                            msdict[dataset]["val_egoleft"]["bev_running"] += val_bev_loss_egoleft
                            msdict[dataset]["val_egoleft"]["reproj_running"] += val_reproj_loss_egoleft

                            # Egoright
                            msdict[dataset]["val_egoright"]["total_running"] += val_total_loss_egoright
                            msdict[dataset]["val_egoright"]["bev_running"] += val_bev_loss_egoright
                            msdict[dataset]["val_egoright"]["reproj_running"] += val_reproj_loss_egoright

                    # Calculate final validation scores for network on each dataset
                    # as well as overall validation score - A lower score is better
                    for dataset in VALID_DATASET_LIST:
                        # Egopath
                        msdict[dataset]["val_egopath"]["total_score"] = msdict[dataset]["val_egopath"]["total_running"] / msdict[dataset]["num_val_samples"]
                        msdict[dataset]["val_egopath"]["bev_score"] = msdict[dataset]["val_egopath"]["bev_running"] / msdict[dataset]["num_val_samples"]
                        msdict[dataset]["val_egopath"]["reproj_score"] = msdict[dataset]["val_egopath"]["reproj_running"] / msdict[dataset]["num_val_samples"]

                        # Egoleft
                        msdict[dataset]["val_egoleft"]["total_score"] = msdict[dataset]["val_egoleft"]["total_running"] / msdict[dataset]["num_val_samples"]
                        msdict[dataset]["val_egoleft"]["bev_score"] = msdict[dataset]["val_egoleft"]["bev_running"] / msdict[dataset]["num_val_samples"]
                        msdict[dataset]["val_egoleft"]["reproj_score"] = msdict[dataset]["val_egoleft"]["reproj_running"] / msdict[dataset]["num_val_samples"]

                        # Egoright
                        msdict[dataset]["val_egoright"]["total_score"] = msdict[dataset]["val_egoright"]["total_running"] / msdict[dataset]["num_val_samples"]
                        msdict[dataset]["val_egoright"]["bev_score"] = msdict[dataset]["val_egoright"]["bev_running"] / msdict[dataset]["num_val_samples"]
                        msdict[dataset]["val_egoright"]["reproj_score"] = msdict[dataset]["val_egoright"]["reproj_running"] / msdict[dataset]["num_val_samples"]

                    # Overall validation metric - total score
                    msdict["overall_val_total_running"] = sum([
                        (
                            msdict[dataset]["val_egopath"]["total_score"] + \
                            msdict[dataset]["val_egoleft"]["total_score"] + \
                            msdict[dataset]["val_egoright"]["total_score"]
                        ) / 3.0

                        for dataset in VALID_DATASET_LIST
                    ])
                    msdict["num_val_overall_samples"] = sum([
                        msdict[dataset]["num_val_samples"]
                        for dataset in VALID_DATASET_LIST
                    ])
                    
                    msdict["overall_val_total_score"] = msdict["val_overall_running"] / msdict["num_val_overall_samples"]

                    # Overall validation metric - bev score
                    msdict["overall_val_bev_running"] = sum([
                        (
                            msdict[dataset]["val_egopath"]["bev_running"] + \
                            msdict[dataset]["val_egoleft"]["bev_running"] + \
                            msdict[dataset]["val_egoright"]["bev_running"]
                        ) / 3.0
                        for dataset in VALID_DATASET_LIST
                    ])
                    msdict["overall_val_bev_score"] = msdict["overall_val_bev_running"] / msdict["num_val_overall_samples"]

                    # Overall validation metric - reproj score
                    msdict["overall_val_reproj_running"] = sum([
                        (
                            msdict[dataset]["val_egopath"]["reproj_running"] + \
                            msdict[dataset]["val_egoleft"]["reproj_running"] + \
                            msdict[dataset]["val_egoright"]["reproj_running"]
                        ) / 3.0
                        for dataset in VALID_DATASET_LIST
                    ])
                    msdict["overall_val_reproj_score"] = msdict["overall_val_reproj_running"] / msdict["num_val_overall_samples"]
                    
                    print("================ Complete - Validation Scores ================")
                    for dataset in VALID_DATASET_LIST:
                        for key in ["egopath", "egoleft", "egoright"]:
                            print(f"\n{dataset} - {key.upper()} TOTAL SCORE : {msdict[dataset]['val_' + key]['total_score']}")
                            print(f"{dataset} - {key.upper()} BEV SCORE : {msdict[dataset]['val_' + key]['bev_score']}")
                            print(f"{dataset} - {key.upper()} REPROJ SCORE : {msdict[dataset]['val_' + key]['reproj_score']}")
                    print("\nOVERALL :")
                    print(f"VAL SCORE : {msdict['overall_val_total_score']}")
                    print(f"VAL BEV SCORE : {msdict['overall_val_bev_score']}")
                    print(f"VAL REPROJ SCORE : {msdict['overall_val_reproj_score']}\n")

                    # Logging average metrics
                    trainer.log_validation(msdict)

                # Switch back to training
                print("================ Continuing with training ================")
                trainer.set_train_mode()
            
            msdict["data_list_count"] = msdict["data_list_count"] + 1


if (__name__ == "__main__"):
    main()