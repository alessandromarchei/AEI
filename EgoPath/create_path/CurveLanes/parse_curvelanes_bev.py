#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import math
import warnings
import numpy as np
from PIL import Image, ImageDraw


# ============================== Helper functions ============================== #


def getLineAnchor(line, img_height):
    """
    Determine "anchor" point of a line, along some related params.
    """
    (x2, y2) = line[0]
    (x1, y1) = line[1]

    for i in range(1, len(line) - 1, 1):
        if (line[i][0] != x2) & (line[i][1] != y2):
            (x1, y1) = line[i]
            break

    if (x1 == x2) or (y1 == y2):
        if (x1 == x2):
            error_lane = "Vertical"
        elif (y1 == y2):
            error_lane = "Horizontal"
        warnings.warn(f"{error_lane} line detected: {line}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2}).")
        return (x1, None, None)
    
    a = (y2 - y1) / (x2 - x1)
    deg = math.degrees(math.atan(-a)) % 180
    b = y1 - a * x1
    x0 = (img_height - b) / a

    return (x0, a, b, deg)