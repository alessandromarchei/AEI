#! /usr/bin/env python3

import os
import json
import torch
import random
import pathlib
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
)))

from argparse import ArgumentParser
from PIL import Image
from typing import Literal, get_args
from Models.data_utils.load_data_ego_path import LoadDataEgoPath
from Models.training.ego_path_trainer import EgoPathTrainer


def main():

    # Argparse init

    parser = ArgumentParser(
        description = "Training module for EgoPath"
    )
    
    parser.add_argument(
        "-s", "--model_save_root_path",
        type = str,
        dest = "model_save_root_path",
        help = "Root path where PyTorch checkpoint save should be saved.",
        required = True
    )
    parser.add_argument(
        "-r", "--root",
        type = str,
        dest = "root_all_datasets",
        help = "Root path where all EgoPath datasets are stored.",
        required = True
    )
    parser.add_argument(
        "-e", "--epochs",
        type = int,
        dest = "num_epochs",
        help = "Number of epochs",
        default = 10,
        required = False
    )
    parser.add_argument(
        "-b", "--batch_size",
        type = int,
        dest = "batch_size",
        help = "Batch size initially (will be gradually reduced via Coarse-to-fine Optimization)",
        default = 32,
        required = False
    )
    args = parser.parse_args()

    root_checkpoints = args.model_save_root_path
    root_datasets = args.root_all_datasets

    # ================== Data acquisition ================== #

    # Data path
    list_datasets = [
        "BDD100K", 
        "COMMA2K19", 
        "CULANE", 
        "CURVELANES", 
        "ROADWORK", 
        "TUSIMPLE"
    ]

    # Master dict to store dataset metadata
    dict_data = {
        dataset : {
            "labels_filepath" : os.path.join(
                root_datasets,
                dataset,
                "drivable_path.json"
            ),
            "images_dirpath" : os.path.join(
                root_datasets,
                dataset,
                "image"
            ),
        }
        for dataset in list_datasets
    }

    # Retrieve em all via LoadDataEgoPath

    for dataset, metadata in dict_data.items():

        # Loader instance
        dict_data[dataset]["loader_instance"] = LoadDataEgoPath(
            labels_filepath = metadata["labels_filepath"],
            images_filepath = metadata["images_dirpath"],
            dataset = dataset
        )

        # Num train/val
        dict_data[dataset]["N_trains"] = dict_data[dataset]["loader_instance"].getItemCount()[0]
        dict_data[dataset]["N_vals"] = dict_data[dataset]["loader_instance"].getItemCount()[1]

    # Count total samples
    SUM_N_TRAINS = sum([
        metadata["N_trains"]
        for _, metadata in dict_data.items()
    ])
    SUM_N_VALS = sum([
        metadata["N_vals"]
        for _, metadata in dict_data.items()
    ])

    # ====================== Training ====================== #

    # Trainer instance
    trainer = EgoPathTrainer()
    trainer.zero_grad()             # Reset optimizer gradients

