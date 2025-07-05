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


def normalizeCoords(line, width, height):
    """
    Normalize the coords of line points.
    """
    return [(x / width, y / height) for x, y in line]


def interpLine(line: list, points_quota: int):
    """
    Interpolates a line of (x, y) points to have at least `point_quota` points.
    This helps with CurveLanes since most of its lines have so few points, 2~3.
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
    (x2, y2) = line[-1]
    (x1, y1) = line[-2]

    for i in range(len(line) - 2, 0, -1):
        if (line[i][0] != x2):
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
    b = y1 - a * x1
    x0 = (H - b) / a
    
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
    bev_egopath: list,
    order: int,
    y_step: int,
    y_limit: int
):
    x = [point[0] for point in bev_egopath]
    y = [point[1] for point in bev_egopath]
    z = np.polyfit(y, x, order)
    f = np.poly1d(z)
    y_new = np.linspace(
        0, y_limit, 
        int(y_limit / y_step) + 1
    )
    x_new = f(y_new)

    # Sort by decreasing y
    fitted_bev_egopath = sorted(
        tuple(zip(x_new, y_new)),
        key = lambda x: x[1],
        reverse = True
    )

    flag_list = [0] * len(fitted_bev_egopath)
    for i in range(len(fitted_bev_egopath)):
        if (not 0 <= fitted_bev_egopath[i][0] <= BEV_W):
            flag_list[i - 1] = 1
            break
    if (not 1 in flag_list):
        flag_list[-1] = 1

    validity_list = [1] * len(fitted_bev_egopath)
    last_valid_index = flag_list.index(1)
    for i in range(last_valid_index + 1, len(validity_list)):
        validity_list[i] = 0
    
    return fitted_bev_egopath, flag_list, validity_list


def imagePointTuplize(point: PointCoords) -> ImagePointCoords:
    """
    Parse all coords of an (x, y) point to int, making it
    suitable for image operations.
    """
    return (int(point[0]), int(point[1]))


def findSourcePointsBEV(
    h: int,
    w: int,
    egoleft: list,
    egoright: list,
) -> dict:
    """
    Find 4 source points for the BEV homography transform.
    """
    sps = {}

    # Renorm 2 egolines
    egoleft = [
        [p[0] * w, p[1] * h]
        for p in egoleft
    ]
    egoright = [
        [p[0] * w, p[1] * h]
        for p in egoright
    ]

    # Acquire LS and RS
    anchor_left = getLineAnchor(egoleft, h)
    anchor_right = getLineAnchor(egoright, h)
    sps["LS"] = [anchor_left[0], h]
    sps["RS"] = [anchor_right[0], h]

    # CALCULATING LE AND RE BASED ON LATEST ALGORITHM

    midanchor_start = [(sps["LS"][0] + sps["RS"][0]) / 2, h]
    ego_height = max(egoleft[-1][1], egoright[-1][1]) * EGO_HEIGHT_LIM

    # Both egos have Null anchors
    if ((not anchor_left[1]) and (not anchor_right[1])):
        midanchor_end = [midanchor_start[0], h]
        original_end_w = sps["RS"][0] - sps["LS"][0]

    else:
        left_deg = 90 if (not anchor_left[1]) else math.degrees(math.atan(anchor_left[1])) % 180
        right_deg = 90 if (not anchor_right[1]) else math.degrees(math.atan(anchor_right[1])) % 180
        mid_deg = (left_deg + right_deg) / 2
        mid_grad = - math.tan(math.radians(mid_deg))
        mid_intercept = h - mid_grad * midanchor_start[0]
        midanchor_end = [
            (ego_height - mid_intercept) / mid_grad,
            ego_height
        ]
        original_end_w = interpX(egoright, ego_height) - interpX(egoleft, ego_height)

    sps["LE"] = [
        midanchor_end[0] - original_end_w / 2,
        ego_height
    ]
    sps["RE"] = [
        midanchor_end[0] + original_end_w / 2,
        ego_height
    ]

    # Tuplize 4 corners
    for i, pt in sps.items():
        sps[i] = imagePointTuplize(pt)

    # Log the ego_height too
    sps["ego_h"] = ego_height

    return sps


def transformBEV(
    img: np.ndarray,
    egopath: list,
    sps: dict
):
    h, w, _ = img.shape

    # Renorm/tuplize drivable path
    egopath = [
        (point[0] * w, point[1] * h) for point in egopath
        if (point[1] * h >= sps["ego_h"])
    ]
    if (not egopath):
        return (None, None, None, None, None)

    # Interp more points for original egopath
    egopath = interpLine(egopath, MIN_POINTS)

    # Get transformation matrix
    mat, _ = cv2.findHomography(
        srcPoints = np.array([
            sps["LS"],
            sps["RS"],
            sps["LE"],
            sps["RE"]
        ]),
        dstPoints = np.array([
            BEV_PTS["LS"],
            BEV_PTS["RS"],
            BEV_PTS["LE"],
            BEV_PTS["RE"],
        ])
    )

    # Transform image
    im_dst = cv2.warpPerspective(
        img, mat,
        np.array([BEV_W, BEV_H])
    )

    # Transform egopath
    bev_egopath = np.array(
        egopath,
        dtype = np.float32
    ).reshape(-1, 1, 2)
    bev_egopath = cv2.perspectiveTransform(bev_egopath, mat)
    bev_egopath = [
        tuple(map(int, point[0])) 
        for point in bev_egopath
    ]

    # Polyfit BEV egopath to get 33-coords format with flags
    bev_egopath, flag_list, validity_list = polyfit_BEV(
        bev_egopath = bev_egopath,
        order = POLYFIT_ORDER,
        y_step = BEV_Y_STEP,
        y_limit = BEV_H
    )

    return (im_dst, bev_egopath, flag_list, validity_list, mat)