import argparse
import os
import json
import cv2
import warnings
from PIL import Image, ImageDraw


# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format

def annotateGT(classified_lanes, gt_images_path, output_dir="visualization"):
    """
    Draws lane lines on images and saves them in the 'visualization' directory.
    
    :param classified_lanes: Dictionary containing lane keypoints.
    :param gt_images_path: Path to the directory containing ground truth images.
    :param output_dir: Directory to save annotated images.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define lane colors
    lane_colors = {
        "egoleft_lane": (0, 255, 0),   # Green
        "egoright_lane": (0, 0, 255),   # Blue
        "other_lanes": (255, 255, 0)    # Yellow
    }
    lane_width = 5

    number_to_store = 0
    
    for image_name, lanes in classified_lanes.items():
        image_path = os.path.join(gt_images_path, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found in {gt_images_path}. Skipping.")
            continue
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Draw each lane
        for lane_type, lane_data in lanes.items():
            color = lane_colors.get(lane_type, (255, 255, 255))  # Default to white if unknown type
            
            for lane in lane_data:
                lane = [(x, y) for x, y in lane]
                draw.line(lane, fill=color, width=lane_width)
        
        if number_to_store <= 50:
            # Save annotated image
            save_path = os.path.join(output_dir, image_name)
            img.save(save_path)
            print(f"Saved annotated image: {save_path}")
            number_to_store += 1
    
    print("Annotation complete. All images saved in", output_dir)

def classify_lanes(normalized_data, IMG_WIDTH, IMG_HEIGHT):
    """
    Classify lanes into ego-left and ego-right based on anchor points.
    """
    result = {}

    for entry in normalized_data:
        image_name = entry["image_name"]
        anchor_points = []

        for lane in entry["lanes"]:
            keypoints = lane["keypoints"][0]  # Extract first set of keypoints
            anchor = getLaneAnchor(keypoints)

            if anchor[0] is not None:  # Avoid lanes with undefined anchors
                anchor_points.append((anchor[0], lane["id"], anchor[1]))


        # Sort lanes by anchor x position
        anchor_points.sort()

        # Determine ego left and right lanes
        ego_indexes = getEgoIndexes(anchor_points)

        if isinstance(ego_indexes, str):
            print(f"Warning: {ego_indexes} in image {image_name}")
            result[image_name] = {
                "egoleft_lane": [],
                "egoright_lane": [],
                "other_lanes": [],
            }
        else:
            left_idx, right_idx = ego_indexes
            result[image_name] = {
                "egoleft_lane": [],
                "egoright_lane": [],
                "other_lanes": [],
            }

            for lane in entry["lanes"]:
                if lane["id"] == anchor_points[left_idx][1]:
                    result[image_name]["egoleft_lane"].append(lane["keypoints"][0])
                elif lane["id"] == anchor_points[right_idx][1]:
                    result[image_name]["egoright_lane"].append(lane["keypoints"][0])
                else:
                    result[image_name]["other_lanes"].append(lane["keypoints"][0])

            # add slope value of lanes.
            # for lane in entry["lanes"]:
            #     if lane["id"] == anchor_points[left_idx][1]:
            #         result[image_name]["egoleft_lane"].append({"keypoints" : lane["keypoints"][0], "slope" : anchor_points[left_idx][2]})
            #     elif lane["id"] == anchor_points[right_idx][1]:
            #         result[image_name]["egoright_lane"].append({"keypoints" : lane["keypoints"][0], "slope" : anchor_points[right_idx][2]})
            #     else:
            #         result[image_name]["other_lanes"].append(lane["keypoints"][0])

    return result


def getLaneAnchor(lane):
    """
    Determine "anchor" point of a lane.

    """
    (x2, y2) = lane[0]
    (x1, y1) = lane[1]
    for i in range(1, len(lane) - 1, 1):
        if lane[i][0] != x2:
            (x1, y1) = lane[i]
            break
    if x1 == x2:
        warnings.warn(
            f"Vertical lane detected: {lane}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2})."
        )
        return (x1, None, None)
    # Get slope and intercept
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    # Get projected point on the image's bottom
    # TODO Handle Horizontal Lanes correctly.
    if a == 0:
        warnings.warn(
            f"Horizontal lane detected: {lane}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2})."
        )
        return (x1, None, None)
    x0 = (IMG_HEIGHT - b) / a

    return (x0, a, b)


def getEgoIndexes(anchors):
    """
    Identifies the two ego lanes (left and right) from sorted anchor points.
    """
    for i in range(len(anchors)):
        if anchors[i][0] >= IMG_WIDTH / 2:
            if i == 0:
                return "NO LANES on the LEFT side of the frame!"
            return (i - 1, i)

    return "NO LANES on the RIGHT side of the frame!"


def normalize_keypoints(lane_train_data, IMG_WIDTH, IMG_HEIGHT):
    normalized_data = []

    # Process each image entry
    for entry in lane_train_data:
        image_name = entry["name"]
        lanes = []
        for label in entry.get("labels", []):
            if "poly2d" in label:
                keypoints = [
                    # [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in poly["vertices"]]
                    [[x, y] for x, y in poly["vertices"]]
                    for poly in label["poly2d"]
                ]

                lanes.append({"id": label["id"], "keypoints": keypoints, "laneDirection": label["attributes"]["laneDirection"]})

        # Store processed data
        normalized_data.append({"image_name": image_name, "lanes": lanes})
    return normalized_data


if __name__ == "__main__":
    # ============================== Dataset structure ============================== #

    # Define Dimension
    IMG_WIDTH = 1280
    IMG_HEIGHT = 720

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description="Process BDD100k dataset - EgoLanes groundtruth generation"
    )

    # bdd100k_train/images/100k/
    parser.add_argument(
        "--image_dir",
        type=str,
        help="BDD100k ground truth image directory (right after extraction)",
    )

    # bdd100k/labels/
    parser.add_argument(
        "--labels_dir",
        type=str,
        help="BDD100k labels directory (right after extraction)",
        required=True,
    )

    parser.add_argument(
        "--output_dir", type=str, help="Desired output directory", required=True
    )

    parser.add_argument(
        "--crop",
        type = int,
        nargs = 4,
        help = "Crop image: [TOP, RIGHT, BOTTOM, LEFT]. Must always be 4 ints. Non-cropped sizes are 0.",
        metavar = ("TOP", "RIGHT", "BOTTOM", "LEFT"),
        default = [0, 140, 220, 140],
        required = False
    )

    args = parser.parse_args()

    lane_train_json = os.path.join(
        args.labels_dir, "lane", "polygons", "lane_train.json"
    )

    with open(lane_train_json, "r") as f:
        data = json.load(f)

    print("Size of data is ", len(data))

    # Check Data
    # print(json.dumps(data[:1], indent=4))


    # ============================== Crop image ============================== #
    # Parse crop
    if (args.crop):
        print(f"Cropping image set with sizes:")
        print(f"\nTOP: {args.crop[0]},\tRIGHT: {args.crop[1]},\tBOTTOM: {args.crop[2]},\tLEFT: {args.crop[3]}")
        if (args.crop[0] + args.crop[2] >=IMG_HEIGHT):
            warnings.warn(f"Cropping size: TOP = {args.crop[0]} and BOTTOM = {args.crop[2]} exceeds image height of {IMG_HEIGHT}. Not cropping.")
            crop = None
        elif (args.crop[1] + args.crop[3] >= IMG_WIDTH):
            warnings.warn(f"Cropping size: LEFT = {args.crop[3]} and RIGHT = {args.crop[1]} exceeds image width of {IMG_HEIGHT}. No cropping.")
            crop = None
        else:
            crop = {
                "TOP" : args.crop[0],
                "RIGHT" : args.crop[1],
                "BOTTOM" : args.crop[2],
                "LEFT" : args.crop[3],
            }
            FORMER_IMG_HEIGHT = IMG_HEIGHT
            FORMER_IMG_WIDTH = IMG_WIDTH
            IMG_HEIGHT -= crop["TOP"] + crop["BOTTOM"]
            IMG_WIDTH -= crop["LEFT"] + crop["RIGHT"]
            print(f"New image size: {IMG_WIDTH}W x {IMG_HEIGHT}H.\n")
    else:
        crop = None

    # ============================== Normalize lane keypoints ============================== #

    normalized_lane_keypoints = normalize_keypoints(data, IMG_WIDTH, IMG_HEIGHT)
    # Save as new JSON file
    # output_file = os.path.join(args.output_dir, "normalized_lanes.json")

    # with open(output_file, "w") as f:
    #     json.dump(normalized_lane_keypoints, f, indent=4)

    # Check Data
    print(
        "Normalized_lane_keypoints_data",
        json.dumps(normalized_lane_keypoints[:1], indent=4),
    )

    # ============================== Save binary mask ============================== #
    # Check output directory exists or not
    # os.makedirs(args.output_dir, exist_ok=True)

    # # Get list of images from input directory
    # image_files = [f for f in os.listdir(os.path.join(args.labels_dir, "lane", "masks", "train")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # binary_lane_mask_path = os.path.join(args.labels_dir, "lane", "masks", "train")
    # print(binary_lane_mask_path)

    # # Process each image
    # for image_name in image_files:
    #     input_path = os.path.join(binary_lane_mask_path, image_name)  # Original image path
    #     output_path = os.path.join(args.output_dir, image_name)  # Keep the same name in output directory

    #     # Read and write the image
    #     img = cv2.imread(input_path)
    #     if img is not None:
    #         cv2.imwrite(output_path, img)
    #         print(f"Saved: {output_path}")
    #     else:
    #         print(f"Skipped: {image_name} (Not a valid image)")

    # ============================== Identify EgoLanes ============================== #
    classified_lanes = classify_lanes(normalized_lane_keypoints, IMG_WIDTH, IMG_HEIGHT)
    # Check Data
    # Assuming classified_lanes is a dictionary
    print("The length of classified_lanes is", len(classified_lanes))
    first_three_items = dict(
        list(classified_lanes.items())[:3]
    )  # Get first 13key-value pairs
    # Pretty print
    print("Classfied_lanes", json.dumps(first_three_items, indent=4))

    # ============================== AnnotateGT ====================================== #

    gt_images_path = os.path.join(args.image_dir, "train")
    annotateGT(classified_lanes, gt_images_path, output_dir=args.output_dir)

