#! /usr/bin/env python3

import os
import json
import pathlib
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
)))
from PIL import Image
from typing import Literal, get_args
from Models.data_utils.check_data import CheckData

VALID_DATASET_LITERALS = Literal[
    # "BDD100K",
    # "COMMA2K19",
    # "CULANE",
    "CURVELANES",
    # "ROADWORK",
    # "TUSIMPLE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))