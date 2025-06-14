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