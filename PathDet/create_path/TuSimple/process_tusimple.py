#! /usr/bin/env python3

import argparse
import json
import os
from PIL import Image, ImageDraw
import warnings

# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"{message}\n"

warnings.formatwarning = custom_warning_format

# ============================== Dataset structure ============================== #

root_dir = "TUSimple"
train_dir = "train_set"
test_dir = "test_set"

train_clip_codes = ["0313", "0531", "0601"] # Train labels are split into 3 dirs
test_file = "test_label.json"               # Test file name

img_width = 1280
img_height = 720

# ============================== Helper functions ============================== #

def normalizeCoords(lane, width, height):

    return [(x / width, y / height) for x, y in lane]


def getLaneAnchor(lane):

    (x2, y2) = lane[-1]
    (x1, y1) = lane[-2]
    for i in range(len(lane) - 2, 0, -1):
        if (lane[i][0] != x2):
            (x1, y1) = lane[i]
            break
    if (x1 == x2):
        warnings.warn(f"Vertical lane detected: {lane}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2}).")
        return (x1, None, None)
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (img_height - b) / a
    
    return (x0, b)


def getEgoIndexes(anchors):

    # `anchors` is a list of tuples at form (xi, yi)
    # In those labels, lanes are labeled from left to right, so these anchors
    # extracted from them are also sorted x-coords in ascending order.
    for i in range(len(anchors)):
        if (anchors[i][0] >= img_width / 2):
            if (i == 0):
                return "NO LANES on the LEFT side of frame. Something's sussy out there!"
            left_ego_idx, right_ego_idx = i - 1, i
            return (left_ego_idx, right_ego_idx)
    
    return "NO LANES on the RIGHT side of frame. Something's sussy out there!"


def getDrivablePath(left_ego, right_ego):
    i, j = 0, 0
    drivable_path = []
    while (i < len(left_ego) - 1 and j < len(right_ego) - 1):
        if (left_ego[i][1] == right_ego[j][1]):
            drivable_path.append((
                (left_ego[i][0] + right_ego[j][0]) / 2,     # Midpoint along x axis
                left_ego[i][1]
            ))
            i += 1
            j += 1
        elif (left_ego[i][1] < right_ego[j][1]):
            i += 1
        else:
            j += 1

    return drivable_path


def annotateGT(
        anno_entry, anno_raw_file, 
        raw_dir, labeled_dir, 
        normalized = True
):

    # Load raw image
    raw_img = Image.open(anno_raw_file).convert("RGB")

    # Copy raw img and put it in raw dir.
    # Keep original pathname for traceability, but replace "/" with "-"
    raw_img.save(os.path.join(raw_dir, str(anno_raw_file).replace("/", "-")))
    
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
            lane = [(x * img_width, y * img_height) for x, y in lane]
        if (idx in anno_entry["ego_indexes"]):
            # Ego lanes, in green
            draw.line(lane, fill = lane_colors["ego_green"], width = lane_w)
        else:
            # Outer lanes, in red
            draw.line(lane, fill = lane_colors["outer_red"], width = lane_w)
    # Drivable path, in yellow
    if (normalized):
        anno_entry["drivable_path"] = [(x * img_width, y * img_height) for x, y in anno_entry["drivable_path"]]
    draw.line(anno_entry["drivable_path"], fill = lane_colors["drive_path_yellow"], width = lane_w)

    # Save labeled img, same format with raw, just different dir
    raw_img.save(os.path.join(labeled_dir, str(anno_raw_file).replace('/', '-')))


def parseAnnotations(anno_path):

    # Read em raw
    with open(anno_path, "r") as f:
        read_data = [json.loads(line) for line in f.readlines()]

    # Parse em 
    anno_data = {}
    for item in read_data:
        # Read raw data
        lanes = item["lanes"]
        h_samples = item["h_samples"]
        raw_file = item["raw_file"]

        # Decouple from {lanes: [xi1, xi2, ...], h_samples: [y1, y2, ...]} to [(xi1, y1), (xi2, y2), ...]
        lanes_decoupled = [
            [(x, y) for x, y in zip(lane, h_samples) if x != -2]
            for lane in lanes if sum(1 for x in lane if x != -2) >= 2     # Filter out lanes < 2 points (there's actually a bunch of em)
        ]

        # Determine 2 ego lanes
        lane_anchors = [getLaneAnchor(lane) for lane in lanes_decoupled]
        ego_indexes = getEgoIndexes(lane_anchors)

        if (type(ego_indexes) is str):
            if (ego_indexes.startswith("NO")):
                warnings.warn(f"Parsing {raw_file}: {ego_indexes}")
                continue

        left_ego = lanes_decoupled[ego_indexes[0]]
        right_ego = lanes_decoupled[ego_indexes[1]]

        # Determine drivable path from 2 egos
        drivable_path = getDrivablePath(left_ego, right_ego)

        # Parse processed data, all coords normalized
        anno_data[raw_file] = {
            "lanes": [normalizeCoords(lane, img_width, img_height) for lane in lanes_decoupled],
            "ego_indexes": ego_indexes,
            "drivable_path": normalizeCoords(drivable_path, img_width, img_height),
            "img_size": (img_width, img_height),
        }

    return anno_data

            
if __name__ == "__main__":

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process TuSimple dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "TuSimple directory (right after extraction)"
    )
    parser.add_argument(
        "--output_dir", 
        type = str, 
        help = "Output directory"
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # Generate output structure
    # --train
    #     |
    #     |----raw
    #     |----labeled
    # --test
    #     |
    #     |----raw
    #     |----labeled
    output_dir = args.output_dir
    list_batches = ["train", "test"]
    list_subbatches = ["raw", "labeled"]
    for batch in list_batches:
        for subbatch in list_subbatches:
            sub_dir = os.path.join(output_dir, batch, subbatch)
            if (not os.path.exists(sub_dir)):
                os.makedirs(sub_dir, exist_ok = True)

    # ============================== Parsing annotations ============================== #

    train_label_files = [
        os.path.join(dataset_dir, root_dir, train_dir, f"label_data_{clip_code}.json")
        for clip_code in train_clip_codes
    ]

    """
    Make no mistake: the file `TuSimple/test_set/test_tasks_0627.json` is NOT a label file.
    It is just a template for test submission. Kinda weird they put it in `test_set` while
    the actual label file is in `TuSimple/test_label.json`.
    """
    test_label_files = [os.path.join(dataset_dir, root_dir, test_file)]

    # Parse data by batch
    data_master = {
        "train": {
            "files": train_label_files,
            "data": {}
        },
        "test": {
            "files": test_label_files,
            "data": {}
        },
    }
    for batch in list_batches:
        print(f"\n================================== Processing {batch} data ==================================\n")
        for anno_file in data_master[batch]["files"]:
            print(f"Processing {batch} file {anno_file}...")
            this_data = parseAnnotations(anno_file)
            for raw_file, anno_entry in this_data.items():
                # Annotate raw images
                annotateGT(
                    anno_entry,
                    os.path.join(dataset_dir, root_dir, f"{batch}_set", raw_file), 
                    os.path.join(output_dir, batch, "raw"),
                    os.path.join(output_dir, batch, "labeled")
                )
            data_master[batch]["data"].update(this_data)
            print(f"Processed {len(this_data)} {batch} entries in above file.\n")
        print(f"Done processing {batch} data with {len(data_master[batch]['data'])} entries in total.\n")

    # Save master data
    with open(os.path.join(output_dir, "data_master.json"), "w") as f:
        json.dump(data_master, f, indent = 4)