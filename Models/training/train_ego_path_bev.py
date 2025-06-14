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
from Models.training.ego_path_trainer import EgoPathTrainer

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

    