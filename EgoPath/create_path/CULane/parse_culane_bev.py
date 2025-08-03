#! /usr/bin/env python3

import os
import cv2
import math
import json
import argparse
import warnings
import numpy as np
from PIL import Image, ImageDraw

# Custom warning format
def custom_warning_format(
    message, category, filename, 
    lineno, line = None
):
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


def normalizeCoords(line, width, height):
    """
    Normalize the coords of line points.
    """

    return [
        (x / width, y / height) 
        for x, y in line
    ]


def interpLine(line: list, points_quota: int):
    """
    Interpolates a line of (x, y) points to have at least `point_quota` points.
    """

    if len(line) >= points_quota:
        return line

    # Extract x, y separately then parse to interp
    x = np.array([pt[0] for pt in line])
    y = np.array([pt[1] for pt in line])
    interp_x = np.interp
    interp_y = np.interp

    # Here I try to interp more points along the line, based on
    # distance between each subsequent original points. 

    # 1) Use distance along line as param (t)
    # This is Euclidian distance between each point and the one before it
    distances = np.cumsum(np.sqrt(
        np.diff(x, prepend = x[0])**2 + \
        np.diff(y, prepend = y[0])**2
    ))
    # Force first t as zero
    distances[0] = 0

    # 2) Generate new t evenly spaced along original line
    evenly_t = np.linspace(distances[0], distances[-1], points_quota)

    # 3) Interp x, y coordinates based on evenly t
    x_new = interp_x(evenly_t, distances, x)
    y_new = interp_y(evenly_t, distances, y)

    return list(zip(x_new, y_new))


def getLineAnchor(line):
    """
    Determine "anchor" point of a lane.
    """

    (x2, y2) = line[0]
    (x1, y1) = line[1]

    for i in range(len(line) - 2, 0, -1):
        if (line[i][0] != x2):
            (x1, y1) = line[i]
            break

    if (x1 == x2) or (y1 == y2):
        if (x1 == x2):
            error_lane = "Vertical"
        elif (y1 == y2):
            error_lane = "Horizontal"
        warnings.warn("{0} line detected: {1}, with these 2 anchors: ({2}, {3}), ({4}, {5}).".format(
            error_lane, line, 
            x1, y1, 
            x2, y2
        ))
        return (x1, None, None)
    
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (img_height - b) / a
    
    return (x0, a, b)


def drawLine(
    img: np.ndarray, 
    line: list,
    color: tuple,
    thickness: int = 2
):
    for i in range(1, len(line)):
        pt1 = (
            int(line[i - 1][0]), 
            int(line[i - 1][1])
        )
        pt2 = (
            int(line[i][0]), 
            int(line[i][1])
        )
        cv2.line(
            img, 
            pt1, pt2, 
            color = color, 
            thickness = thickness
        )


def annotateGT(
    img: np.ndarray,
    orig_img: np.ndarray,
    frame_id: str,
    bev_egopath: list,
    reproj_egopath: list,
    bev_egoleft: list,
    reproj_egoleft: list,
    bev_egoright: list,
    reproj_egoright: list,
    raw_dir: str, 
    visualization_dir: str,
    normalized: bool
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Annotated image with all lanes, in "output_dir/visualization".
    """

    # =========================== RAW IMAGE =========================== #

    # Save raw img in raw dir, as PNG
    cv2.imwrite(
        os.path.join(
            raw_dir,
            f"{frame_id}.png"
        ),
        img
    )

    # =========================== BEV VIS =========================== #

    img_bev_vis = img.copy()
    h, w, _ = img_bev_vis.shape

    # Draw egopath
    if (normalized):
        renormed_bev_egopath = [
            (x * w, y * h) 
            for x, y in bev_egopath
        ]
    else:
        renormed_bev_egopath = bev_egopath
    drawLine(
        img = img_bev_vis,
        line = renormed_bev_egopath,
        color = COLOR_EGOPATH
    )
    
    # Draw egoleft
    if (normalized):
        renormed_bev_egoleft = [
            (x * w, y * h) 
            for x, y in bev_egoleft
        ]
    else:
        renormed_bev_egoleft = bev_egoleft
    drawLine(
        img = img_bev_vis,
        line = renormed_bev_egoleft,
        color = COLOR_EGOLEFT
    )

    # Draw egoright
    if (normalized):
        renormed_bev_egoright = [
            (x * w, y * h) 
            for x, y in bev_egoright
        ]
    else:
        renormed_bev_egoright = bev_egoright
    drawLine(
        img = img_bev_vis,
        line = renormed_bev_egoright,
        color = COLOR_EGORIGHT
    )

    # Save visualization img in vis dir, as JPG (saving storage space)
    cv2.imwrite(
        os.path.join(
            visualization_dir,
            f"{frame_id}.jpg"
        ),
        img_bev_vis
    )

    # =========================== ORIGINAL VIS =========================== #

    # Draw reprojected egopath
    if (normalized):
        renormed_reproj_egopath = [
            (x * w, y * h) 
            for x, y in reproj_egopath
        ]
    else:
        renormed_reproj_egopath = reproj_egopath
    drawLine(
        img = orig_img,
        line = renormed_reproj_egopath,
        color = COLOR_EGOPATH
    )
    
    # Draw reprojected egoleft
    if (normalized):
        renormed_reproj_egoleft = [
            (x * w, y * h) 
            for x, y in reproj_egoleft
        ]
    else:
        renormed_reproj_egoleft = reproj_egoleft
    drawLine(
        img = orig_img,
        line = renormed_reproj_egoleft,
        color = COLOR_EGOLEFT
    )

    # Draw reprojected egoright
    if (normalized):
        renormed_reproj_egoright = [
            (x * w, y * h) 
            for x, y in reproj_egoright
        ]
    else:
        renormed_reproj_egoright = reproj_egoright
    drawLine(
        img = orig_img,
        line = renormed_reproj_egoright,
        color = COLOR_EGORIGHT
    )

    # Save it
    cv2.imwrite(
        os.path.join(
            visualization_dir,
            f"{frame_id}_orig.jpg"
        ),
        orig_img
    )


def interpX(line, y):
    """
    Interpolate x-value of a point on a line, given y-value
    """

    points = np.array(line)
    list_x = points[:, 0]
    list_y = points[:, 1]

    if not np.all(np.diff(list_y) > 0):
        sort_idx = np.argsort(list_y)
        list_y = list_y[sort_idx]
        list_x = list_x[sort_idx]

    return float(np.interp(y, list_y, list_x))


def polyfit_BEV(
    bev_line: list,
    order: int,
    y_step: int,
    y_limit: int
):
    valid_line = [
        point for point in bev_line
        if (0 <= point[0] < BEV_W) and (0 <= point[1] < BEV_H)
    ]
    x = [point[0] for point in valid_line]
    y = [point[1] for point in valid_line]
    z = np.polyfit(y, x, order)
    f = np.poly1d(z)
    y_new = np.linspace(
        0, y_limit, 
        int(y_limit / y_step) + 1
    )
    x_new = f(y_new)

    # Sort by decreasing y
    fitted_bev_line = sorted(
        tuple(zip(x_new, y_new)),
        key = lambda x: x[1],
        reverse = True
    )

    flag_list = [0] * len(fitted_bev_line)
    for i in range(len(fitted_bev_line)):
        if (not 0 <= fitted_bev_line[i][0] <= BEV_W):
            flag_list[i - 1] = 1
            break
    if (not 1 in flag_list):
        flag_list[-1] = 1

    validity_list = [1] * len(fitted_bev_line)
    last_valid_index = flag_list.index(1)
    for i in range(last_valid_index + 1, len(validity_list)):
        validity_list[i] = 0
    
    return fitted_bev_line, flag_list, validity_list