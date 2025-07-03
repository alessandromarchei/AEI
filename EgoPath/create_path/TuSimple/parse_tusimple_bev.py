#! /usr/bin/env python3

import os
import cv2
import math
import json
import argparse
import warnings
import numpy as np
from PIL import Image, ImageDraw

# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format

PointCoords = tuple[float, float]
ImagePointCoords = tuple[int, int]

# Skipped frames
skipped_dict = {}

# ============================== Helper functions ============================== #


def log_skipped(frame_id, reason):
    skipped_dict[frame_id] = reason

def roundLineFloats(line, ndigits = 4):
    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)
    return line

