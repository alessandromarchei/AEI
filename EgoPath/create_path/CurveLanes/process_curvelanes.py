#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import pathlib
from PIL import Image, ImageDraw
import warnings
from datetime import datetime
import numpy as np

# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format

# ============================== Helper functions ============================== #

def normalizeCoords(lane, width, height):
    """
    Normalize the coords of lane points.
    """
    return [(x / width, y / height) for x, y in lane]


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
                return "NO LANES on the LEFT side of frame. Something's sussy out there!"
            left_ego_idx, right_ego_idx = i - 1, i
            return (left_ego_idx, right_ego_idx)

    return "NO LANES on the RIGHT side of frame. Something's sussy out there!"


def getDrivablePath(left_ego, right_ego, y_coords_interp = False):
    """
    Computes drivable path as midpoint between 2 ego lanes, basically the main point of this task.
    """
    print(f"Left ego: {left_ego}")
    print(f"Right ego: {right_ego}")
    i, j = 0, 0
    drivable_path = []

    # When shit hits the fan and we need interpolation among non-uniform y-coords
    if (y_coords_interp):
        left_ego = np.array(left_ego)
        right_ego = np.array(right_ego)
        y_coords_ASSEMBLE = np.unique(
            np.concatenate((
                left_ego[:, 1],
                right_ego[:, 1]
            ))
        )[::-1]
        print(y_coords_ASSEMBLE)
        left_x_interp = np.interp(y_coords_ASSEMBLE, left_ego[:, 1][::-1], left_ego[:, 0][::-1])
        print(f"Left x interp: {left_x_interp}")
        right_x_interp = np.interp(y_coords_ASSEMBLE, right_ego[:, 1][::-1], right_ego[:, 0][::-1])
        print(f"Right x interp: {right_x_interp}")
        mid_x = (left_x_interp + right_x_interp) / 2
        drivable_path = list(zip(mid_x, y_coords_ASSEMBLE))
    else:
        # Get the normal drivable path from the longest common y-coords
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


def annotateGT(
        raw_img, anno_entry,
        raw_dir, visualization_dir, mask_dir,
        init_img_width, init_img_height,
        normalized = True,
        resize = None,
        crop = None,
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Annotated image with all lanes, in "output_dir/visualization".
        - Binary segmentation mask of drivable path, in "output_dir/segmentation".
    """

    # Load img
    raw_img = raw_img
    new_img_height = init_img_height
    new_img_width = init_img_width
    
    # Handle image resizing
    if (resize):
        raw_img = raw_img.resize((
            int(new_img_height * resize), 
            int(new_img_height * resize)
        ))
        new_img_height *= resize
        new_img_width *= resize

    # Handle image cropping
    if (crop):
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]
        raw_img = raw_img.crop((
            CROP_LEFT, 
            CROP_TOP, 
            init_img_width - CROP_RIGHT, 
            init_img_height - CROP_BOTTOM
        ))
        new_img_height -= CROP_TOP + CROP_BOTTOM
        new_img_width -= CROP_LEFT + CROP_RIGHT

    # Define save name
    # Also save in PNG (EXTREMELY SLOW compared to jpg, for lossless quality)
    save_name = str(img_id_counter).zfill(6) + ".jpg"

    # Copy raw img and put it in raw dir.
    raw_img.save(os.path.join(raw_dir, save_name))

    # Draw all lanes & lines
    draw = ImageDraw.Draw(raw_img)
    lane_colors = {
        "outer_red": (255, 0, 0), 
        "ego_green": (0, 255, 0), 
        "drive_path_yellow": (255, 255, 0)
    }
    lane_w = 5
    # Draw lanes
    for idx, lane in enumerate(anno_entry["lanes"]):
        if (normalized):
            lane = [
                (x * new_img_width, y * new_img_height) 
                for x, y in lane
            ]
        if (idx in anno_entry["ego_indexes"]):
            # Ego lanes, in green
            draw.line(lane, fill = lane_colors["ego_green"], width = lane_w)
        else:
            # Outer lanes, in red
            draw.line(lane, fill = lane_colors["outer_red"], width = lane_w)
    # Drivable path, in yellow
    if (normalized):
        drivable_renormed = [
            (x * new_img_width, y * new_img_height) 
            for x, y in anno_entry["drivable_path"]
        ]
    else:
        drivable_renormed = anno_entry["drivable_path"]
    draw.line(drivable_renormed, fill = lane_colors["drive_path_yellow"], width = lane_w)

    # Save visualization img, same format with raw, just different dir
    raw_img.save(os.path.join(visualization_dir, save_name))

    # Working on binary mask
    mask = Image.new("L", (new_img_width, new_img_height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.line(drivable_renormed, fill = 255, width = lane_w)
    mask.save(os.path.join(mask_dir, save_name))


def parseAnnotations(
        anno_path, 
        init_img_width,
        init_img_height,
        crop = None,
        resize = None,
    ):
    """
    Parses lane annotations from raw img + anno files, then extracts normalized GT data.
    """
    # Read each line of GT text file as JSON
    with open(anno_path, "r") as f:
        read_data = json.load(f)["Lines"]
        if (len(read_data) < 2):    # Some files are empty, or having less than 2 lines
            warnings.warn(f"Parsing {anno_path} : insufficient lane amount: {len(read_data)}")
            return None
        else:
            # Parse data from those JSON lines
            lanes = [
                [(float(point["x"]), float(point["y"])) for point in line]
                for line in read_data
            ]

            new_img_height = init_img_height
            new_img_width = init_img_width

            # Handle image resizing
            if (resize):
                new_img_height *= resize
                new_img_width *= resize
                lanes = [[
                    (x * resize, y * resize) 
                    for (x, y) in lane
                ] for lane in lanes]

            # Handle image cropping
            if (crop):
                CROP_TOP = crop["TOP"]
                CROP_RIGHT = crop["RIGHT"]
                CROP_BOTTOM = crop["BOTTOM"]
                CROP_LEFT = crop["LEFT"]
                # Crop
                new_img_height -= CROP_TOP + CROP_BOTTOM
                new_img_width -= CROP_LEFT + CROP_RIGHT
                lanes = [[
                    (x - CROP_LEFT, y - CROP_TOP) for x, y in lane
                    if (CROP_LEFT <= x <= (init_img_width - CROP_RIGHT)) and (CROP_TOP <= y <= (init_img_height - CROP_BOTTOM))
                ] for lane in lanes]
                # Remove empty lanes
                lanes = [lane for lane in lanes if (lane and len(lane) >= 2)]   # Pick lanes with >= 2 points
                if (len(lanes) < 2):    # Ignore frames with less than 2 lanes
                    warnings.warn(f"Parsing {anno_path}: insufficient lane amount after cropping: {len(lanes)}")
                    return None

            # Determine 2 ego lanes
            lane_anchors = [getLaneAnchor(lane) for lane in lanes]
            ego_indexes = getEgoIndexes(lane_anchors)

            if (type(ego_indexes) is str):
                if (ego_indexes.startswith("NO")):
                    warnings.warn(f"Parsing {anno_path}: {ego_indexes}")
                return None

            left_ego = lanes[ego_indexes[0]]
            right_ego = lanes[ego_indexes[1]]

            # Determine drivable path from 2 egos, and switch on interp cuz this is CurveLanes
            drivable_path = getDrivablePath(
                left_ego, right_ego,
                y_coords_interp = True
            )

            # Parse processed data, all coords normalized
            anno_data = {
                "lanes" : [normalizeCoords(lane, new_img_width, new_img_height) for lane in lanes],
                "ego_indexes" : ego_indexes,
                "drivable_path" : normalizeCoords(drivable_path, new_img_width, new_img_height),
                "img_width" : new_img_width,
                "img_height" : new_img_height,
            }

            return anno_data


if __name__ == "__main__":

    # ============================== Dataset structure ============================== #

    root_dir = "Curvelanes"
    list_splits = ["train", "valid"]
    img_dir = "images"
    label_dir = "labels"

    # I got this result from `./EDA_imgsizes.ipynb`
    size_dict = {
        "beeg" : [2560, 1440],
        "half_beeg" : [1280, 720],
        "weird" : [1570, 660],
    }
    beeg_y_crop = 240
    beeg_x_crop = 160
    weird_y_crop = 130
    weird_x_crop = 385

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process CurveLanes dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "CurveLanes directory (should contain exactly `Curvelanes` if you get it from Kaggle)",
        required = True
    )
    parser.add_argument(
        "--output_dir", 
        type = str, 
        help = "Output directory",
        required = True
    )
    # parser.add_argument(
    #     "--crop",
    #     type = int,
    #     nargs = 4,
    #     help = "Crop image: [TOP, RIGHT, BOTTOM, LEFT]. Must always be 4 ints. Non-cropped sizes are 0.",
    #     metavar = ("TOP", "RIGHT", "BOTTOM", "LEFT"),
    #     default = [0, 390, 160, 390],    # 2 plus 2 is 4 minus 1 that's 3, quick maths
    #     required = False
    # )
    parser.add_argument(
        "--sampling_step",
        type = int,
        help = "Sampling step for each split/class",
        required = False,
        default = 5
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        type = int,
        help = "Num. files each split/class you wanna limit, instead of whole set.",
        required = False
    )
    args = parser.parse_args()

    # Parse dirs
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # Parse sampling step
    if (args.sampling_step):
        print(f"Sampling step set to {args.sampling_step}.")
        sampling_step = args.sampling_step
    else:
        sampling_step = 1

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, each split/class stops after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Generate output structure
    """
    --output_dir
        |----image
        |----segmentation
        |----visualization
        |----drivable_path.json
    """
    list_subdirs = ["image", "segmentation", "visualization"]
    if (os.path.exists(output_dir)):
        warnings.warn(f"Output directory {output_dir} already exists. Purged")
        shutil.rmtree(output_dir)
    for subdir in list_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if (not os.path.exists(subdir_path)):
            os.makedirs(subdir_path, exist_ok = True)

    # ============================== Parsing annotations ============================== #

    # Parse data by batch
    data_master = {}

    for split in list_splits:
        print(f"\n==================== Processing {split} data ====================\n")
        img_id_counter = -1
        raw_img_book = os.path.join(dataset_dir, root_dir, split, f"{split}.txt")
        with open(raw_img_book, "r") as f:
            list_raw_files = f.readlines()

            for i in range(0, len(list_raw_files), sampling_step):
                img_path = os.path.join(dataset_dir, root_dir, split, list_raw_files[i]).strip()
                img_id_counter += 1
                # Preload image file for multiple uses later
                raw_img = Image.open(img_path).convert("RGB")
                img_width, img_height = raw_img.size
                # if (img_size == size_dict["beeg"]):
                #     resize = 0.5
                #     crop = {
                #         "TOP" : beeg_y_crop,
                #         "RIGHT" : beeg_x_crop,
                #         "BOTTOM" : beeg_y_crop,
                #         "LEFT" : beeg_x_crop
                #     }
                # elif (img_size == size_dict["half_beeg"]):
                #     resize = None
                #     crop = {
                #         "TOP" : beeg_y_crop,
                #         "RIGHT" : beeg_x_crop,
                #         "BOTTOM" : beeg_y_crop,
                #         "LEFT" : beeg_x_crop
                #     }
                # elif (img_size == size_dict["weird"]):
                #     resize = None
                #     crop = {
                #         "TOP" : weird_y_crop,
                #         "RIGHT" : weird_x_crop,
                #         "BOTTOM" : weird_y_crop,
                #         "LEFT" : weird_x_crop
                #     }
                resize = None
                crop = None

                anno_path = img_path.replace(".jpg", ".lines.json").replace(img_dir, label_dir)

                this_data = parseAnnotations(
                    anno_path = anno_path,
                    init_img_width = img_width,
                    init_img_height = img_height,
                    resize = resize,
                    crop = crop
                )
                print(this_data)
                if (this_data is not None):

                    print(f"Processing data in label file {anno_path}.")
                    annotateGT(
                        raw_img = raw_img,
                        anno_entry = this_data,
                        raw_dir = os.path.join(output_dir, "image"),
                        visualization_dir = os.path.join(output_dir, "visualization"),
                        mask_dir = os.path.join(output_dir, "segmentation"),
                        init_img_height = img_height,
                        init_img_width = img_width,
                        resize = resize,
                        crop = crop
                    )

                    # Save as 6-digit incremental index
                    img_index = str(str(img_id_counter).zfill(6))
                    data_master[img_index] = {}
                    data_master[img_index]["drivable_path"] = this_data["drivable_path"]
                    data_master[img_index]["img_height"] = this_data["img_height"]
                    data_master[img_index]["img_width"] = this_data["img_width"]

                    # Early stopping, it defined
                    if (early_stopping and img_id_counter >= early_stopping - 1):
                        break

    # Save master data
    with open(os.path.join(output_dir, "drivable_path.json"), "w") as f:
        json.dump(data_master, f, indent = 4)