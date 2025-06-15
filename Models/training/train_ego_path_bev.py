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
    dict_datasets = {}
    for dataset in VALID_DATASET_LIST:
        dict_datasets[dataset] = {
            "path_labels" : os.path.join(ROOT_PATH, dataset, BEV_JSON_PATH),
            "path_images" : os.path.join(ROOT_PATH, dataset, BEV_IMG_PATH)
        }

    # Deal with TEST dataset
    dict_datasets["TEST"] = {
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
            labels_filepath = dict_datasets[dataset]["path_labels"],
            images_filepath = dict_datasets[dataset]["path_images"],
            dataset = dataset
        )
        N_trains, N_vals = this_dataset_loader.getItemCount()
        random_sample_list = random.shuffle(list(range(0, N_trains)))

        dict_datasets[dataset]["loader"] = this_dataset_loader
        dict_datasets[dataset]["N_trains"] = N_trains
        dict_datasets[dataset]["N_vals"] = N_vals
        dict_datasets[dataset]["sample_list"] = random_sample_list

        print(f"LOADED: {dataset} with {N_trains} train samples, {N_vals} val samples.")

    # All datasets - stats

    dict_datasets["Nsum_trains"] = sum([
        dict_datasets[dataset]["N_trains"]
        for dataset in VALID_DATASET_LIST
    ])
    print(f"Total train samples: {dict_datasets['Nsum_trains']}")

    dict_datasets["Nsum_vals"] = sum([
        dict_datasets[dataset]["N_vals"]
        for dataset in VALID_DATASET_LIST
    ])
    print(f"Total val samples: {dict_datasets['Nsum_vals']}")

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
        dict_datasets["data_list_count"] = 0
        
        # Reset all data counters
        dict_datasets["sample_counter"] = 0
        for dataset in VALID_DATASET_LIST:
            dict_datasets[dataset]["iter"] = 0
            dict_datasets[dataset]["completed"] = False

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
            dict_datasets["sample_counter"] += 1
            dict_datasets["log_counter"] = (
                dict_datasets["sample_counter"] + \
                dict_datasets["Nsum_trains"] * epoch
            )

            # Reset iterators and shuffle individual datasets
            # based on data sampling scheme
            for dataset in VALID_DATASET_LIST:
                if (dict_datasets[dataset]["iter"] == dict_datasets[dataset]["N_trains"]):
                    if (DATA_SAMPLING_SCHEME == "EQUAL"):
                        dict_datasets[dataset]["iter"] = 0
                        random.shuffle(dict_datasets[dataset]["sample_list"])
                    elif (
                        (DATA_SAMPLING_SCHEME == "CONCATENATE") and 
                        (dict_datasets[dataset]["completed"] == False)
                    ):
                        data_list.remove(dataset)

                    dict_datasets[dataset]["completed"] = True

            # If we have looped through each dataset at least once - restart the epoch
            if (all([
                dict_datasets[dataset]["completed"]
                for dataset in VALID_DATASET_LIST
            ])):
                break

            # Reset the data list count if out of range
            if (dict_datasets["data_list_count"] >= len(data_list)):
                dict_datasets["data_list_count"] = 0

            # Fetch data from current processed dataset
            
            image = None
            xs = []
            ys = []
            flags = None

            current_dataset = data_list[dict_datasets["data_list_count"]]
            current_dataset_iter = dict_datasets[current_dataset]["iter"]
            image, xs, ys, flags = dict_datasets[current_dataset]["loader"].getItem(
                dict_datasets[current_dataset]["sample_list"][current_dataset_iter],
                is_train = True
            )
            current_dataset_iter += 1

            # Start the training on this data

            