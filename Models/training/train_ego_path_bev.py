#! /usr/bin/env python3

import os
import torch
import random
import pathlib
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from Models.data_utils.load_data_ego_path_bev import LoadDataBEVEgoPath
from Models.training.ego_path_trainer import EgoPathTrainer


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

    