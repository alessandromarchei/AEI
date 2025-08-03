#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import pathlib
from PIL import Image, ImageDraw
import warnings
from datetime import datetime

# Custom warning format
def custom_warning_format(
    message, category, filename, 
    lineno, line = None
):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format


# ============================== Helper functions ============================== #


def roundLineFloats(line, ndigits = 4):
    """
    Round floats to reduce JSON size.

    """
    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)
    return line


def normalizeCoords(lane, width, height):
    """
    Normalize the coords of lane points.

    """
    return [
        (x / width, y / height) 
        for x, y in lane
    ]


