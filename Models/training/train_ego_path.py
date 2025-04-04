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
    args = parser.parse_args()

    root_checkpoints = args.model_save_root_path
    root_datasets = args.root_all_datasets

    