#! /usr/bin/env python3

import os
import cv2
import math
import json
import argparse
import warnings
import numpy as np
from PIL import Image, ImageDraw
from .process_curvelanes import custom_warning_format, round_line_floats

warnings.formatwarning = custom_warning_format

PointCoords = tuple[float, float]
ImagePointCoords = tuple[int, int]

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


def interpMorePoints(
    line: list,
    min_points: int
):
    if (len(line) < min_points):
        new_line = []
        for i in range(len(line) - 1):
            x0, y0 = line[i]
            x1, y1 = line[i + 1]
            
            new_line.append((x0, y0))  # include starting point

            # Interpolate `min_points` points between (x0, y0) and (x1, y1)
            for j in range(1, min_points + 1):
                t = j / (min_points + 1)
                x = x0 + t * (x1 - x0)
                y = y0 + t * (y1 - y0)
                new_line.append((x, y))

        new_line.append(line[-1])  # include the last point
        return new_line
    
    return line


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
    middeg = (anchor_left[3] + anchor_right[3]) / 2
    mid_grad = - math.tan(math.radians(middeg))
    mid_intercept = h - mid_grad * midanchor_start[0]

    ego_height = max(egoleft[-1][1], egoright[-1][1]) * 1.05
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
    for i, pt in enumerate(sps):
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

    # Interp more points for original egopath
    egopath = interpMorePoints(egopath, MIN_POINTS)

    # Get transformation matrix
    mat, _ = cv2.findHomography(
        srcPoints = np.array([
            sps["LS"],
            sps["RS"],
            sps["LE"],
            sps["RE"]
        ]),
        dstPoints = np.array([
            BEV_pts["LS"],
            BEV_pts["RS"],
            BEV_pts["LE"],
            BEV_pts["RE"],
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

    return im_dst, bev_egopath, mat


# ============================== Main run ============================== #


if __name__ == "__main__":

    # DIRECTORY STRUCTURE

    IMG_DIR = "image"
    JSON_PATH = "drivable_path.json"

    BEV_IMG_DIR = "image_bev"
    BEV_VIS_DIR = "visualization_bev"
    BEV_JSON_PATH = "drivable_path_bev.json"

    # OTHER PARAMS

    MIN_POINTS = 30

    BEV_pts = {
        "LS" : [120, 640],  # Left start
        "RS" : [200, 640],  # Right start
        "LE" : [120, 0],    # Left end
        "RE" : [200, 0]     # Right end
    }

    BEV_W = 320
    BEV_H = 640
    BEV_Y_STEP = 20

    # PARSING ARGS

    parser = argparse.ArgumentParser(
        description = "Generating BEV from CurveLanes processed datasets"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "Processed CurveLanes directory",
        required = True
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        type = int,
        help = "Num. frames you wanna limit, instead of whole set.",
        required = False
    )
    args = parser.parse_args()

    # Parse dataset dir
    dataset_dir = args.dataset_dir
    IMG_DIR = os.path.join(dataset_dir, IMG_DIR)
    JSON_PATH = os.path.join(dataset_dir, JSON_PATH)
    BEV_JSON_PATH = os.path.join(dataset_dir, BEV_JSON_PATH)

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, each split/class stops after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Generate new dirs and paths
    BEV_IMG_DIR = os.path.join(dataset_dir, BEV_IMG_DIR)
    BEV_VIS_DIR = os.path.join(dataset_dir, BEV_VIS_DIR)

    if not (os.path.exists(BEV_IMG_DIR)):
        os.makedirs(BEV_IMG_DIR)
    if not (os.path.exists(BEV_VIS_DIR)):
        os.makedirs(BEV_VIS_DIR)

    # Preparing data
    with open(JSON_PATH, "r") as f:
        json_data = json.load(f)
    data_master = {}    # Dumped later

    # MAIN GENERATION LOOP

    for frame_id, frame_content in enumerate(json_data):

        # Acquire frame
        frame_img_path = os.path.join(
            IMG_DIR,
            f"{frame_id}.png"
        )
        img = cv2.imread(frame_img_path)
        h, w, _ = img.shape

        # Acquire frame data
        this_frame_data = json_data[frame_id]

        # Get source points for transform
        sps_dict = findSourcePointsBEV(
            h = h,
            w = w,
            egoleft = this_frame_data["egoleft_lane"],
            egoright = this_frame_data["egoright_lane"]
        )

        # Transform to BEV space
        im_dst, bev_egopath, mat = transformBEV(
            img = img,
            egopath = this_frame_data["drivable_path"],
            sps = sps_dict
        )