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


def getLaneAnchor(lane):
    """
    Determine "anchor" point of a lane.

    """
    (x2, y2) = lane[0]
    (x1, y1) = lane[1]
    for i in range(1, len(lane) - 1, 1):
        if (lane[i][0] != x2):
            (x1, y1) = lane[i]
            break
    if (x1 == x2):
        warnings.warn(f"Vertical lane detected: {lane}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2}).")
        return (x1, None, None)
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (img_height - b) / a
    
    return (x0, a, b)


def getEgoIndexes(anchors):
    """
    Identifies 2 ego lanes - left and right - from a sorted list of lane anchors.

    """
    for i in range(len(anchors)):
        if (anchors[i][0] >= img_width / 2):
            if (i == 0):
                return "NO LANES on the LEFT side of frame."
            left_ego_idx, right_ego_idx = i - 1, i
            return (left_ego_idx, right_ego_idx)
    
    return "NO LANES on the RIGHT side of frame."


def getDrivablePath(left_ego, right_ego):
    """
    Computes drivable path as midpoint between 2 ego lanes, basically the main point of this task.

    """
    i, j = 0, 0
    drivable_path = []
    while (i <= len(left_ego) - 1 and j <= len(right_ego) - 1):
        if (left_ego[i][1] == right_ego[j][1]):
            drivable_path.append((
                (left_ego[i][0] + right_ego[j][0]) / 2,     # Midpoint along x axis
                left_ego[i][1]
            ))
            i += 1
            j += 1
        elif (left_ego[i][1] > right_ego[j][1]):
            i += 1
        else:
            j += 1

    # Extend drivable path to bottom edge of the frame
    if ((len(drivable_path) >= 2) and (drivable_path[0][1] < img_height)):
        x1, y1 = drivable_path[1]
        x2, y2 = drivable_path[0]
        if (x2 == x1):
            x_bottom = x2
        else:
            a = (y2 - y1) / (x2 - x1)
            x_bottom = x2 + (img_height - y2) / a
        drivable_path.insert(0, (x_bottom, img_height))

    # Extend drivable path to be on par with longest ego lane
    # By making it parallel with longer ego lane
    y_top = min(left_ego[-1][1], right_ego[-1][1])
    if ((len(drivable_path) >= 2) and (drivable_path[-1][1] > y_top)):
        sign_left_ego = left_ego[-1][0] - left_ego[-2][0]
        sign_right_ego = right_ego[-1][0] - right_ego[-2][0]
        sign_val = sign_left_ego * sign_right_ego
        # 2 egos going the same direction
        if (sign_val > 0):
            longer_ego = left_ego if left_ego[-1][1] < right_ego[-1][1] else right_ego
            if len(longer_ego) >= 2 and len(drivable_path) >= 2:
                x1, y1 = longer_ego[-1]
                x2, y2 = longer_ego[-2]
                if (x2 == x1):
                    x_top = drivable_path[-1][0]
                else:
                    a = (y2 - y1) / (x2 - x1)
                    x_top = drivable_path[-1][0] + (y_top - drivable_path[-1][1]) / a

                drivable_path.append((x_top, y_top))
        # 2 egos going opposite directions
        else:
            if len(drivable_path) >= 2:
                x1, y1 = drivable_path[-1]
                x2, y2 = drivable_path[-2]
                if (x2 == x1):
                    x_top = x1
                else:
                    a = (y2 - y1) / (x2 - x1)
                    x_top = x1 + (y_top - y1) / a

                drivable_path.append((x_top, y_top))

    return drivable_path