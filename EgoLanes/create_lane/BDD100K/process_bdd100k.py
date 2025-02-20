import argparse
import os
import json
import cv2
import warnings
from PIL import Image, ImageDraw
import copy


# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"


warnings.formatwarning = custom_warning_format


def normalize_and_crop(data, args):
    # Parse crop
    if args.crop:
        print(f"Cropping image set with sizes:")
        print(
            f"\nTOP: {args.crop[0]},\tRIGHT: {args.crop[1]},\tBOTTOM: {args.crop[2]},\tLEFT: {args.crop[3]}"
        )
        if args.crop[0] + args.crop[2] >= ORIGINAL_IMG_HEIGHT:
            warnings.warn(
                f"Cropping size: TOP = {args.crop[0]} and BOTTOM = {args.crop[2]} exceeds image height of {ORIGINAL_IMG_HEIGHT}. Not cropping."
            )
            crop = None
        elif args.crop[1] + args.crop[3] >= ORIGINAL_IMG_WIDTH:
            warnings.warn(
                f"Cropping size: LEFT = {args.crop[3]} and RIGHT = {args.crop[1]} exceeds image width of {ORIGINAL_IMG_WIDTH}. No cropping."
            )
            crop = None
        else:
            crop = {
                "TOP": args.crop[0],
                "RIGHT": args.crop[1],
                "BOTTOM": args.crop[2],
                "LEFT": args.crop[3],
            }
            print(f"New image size: {IMG_WIDTH}W x {IMG_HEIGHT}H.\n")
    else:
        crop = None

    result = copy.deepcopy(data)

    if crop:
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]

        # Edit each entry
        for entry in result:
            image_name = entry["name"]
            lanes = []
            for label in entry.get("labels", []):
                if "poly2d" in label:
                    for poly in label["poly2d"]:
                        new_vertices = []
                        for x, y in poly["vertices"]:
                            if (
                                CROP_LEFT <= x <= (ORIGINAL_IMG_WIDTH - CROP_RIGHT)
                            ) and (
                                CROP_TOP <= y <= (ORIGINAL_IMG_HEIGHT - CROP_BOTTOM)
                            ):
                                new_x = x - CROP_LEFT
                                new_y = y - CROP_TOP
                                # Normalize and crop
                                new_vertices.append(
                                    [new_x / IMG_WIDTH, new_y / IMG_HEIGHT]
                                )

                        # Update the list in place
                        if len(new_vertices) >= 2:
                            poly["vertices"] = new_vertices
                        else:
                            poly["vertices"] = []
                            warnings.warn(
                                f"Parsing {image_name} : Insufficient lane amount after cropping: {len(lanes)}"
                            )
        # Print all the crop items
        print(f"Cropping image set with sizes:")
        print(
            f"\nTOP: {CROP_TOP},\tRIGHT: {CROP_RIGHT},\tBOTTOM: {CROP_BOTTOM},\tLEFT: {CROP_LEFT},\tORIGINAL_IMG_WIDTH: {ORIGINAL_IMG_WIDTH},\tORIGINAL_IMG_HEIGHT: {ORIGINAL_IMG_HEIGHT}"
        )
    return result


def annotateGT(classified_lanes, gt_images_path, output_dir="visualization", crop=None):
    """
    Draws lane lines on images and saves them in the 'visualization' directory.

    :param classified_lanes: Dictionary containing lane keypoints.
    :param gt_images_path: Path to the directory containing ground truth images.
    :param output_dir: Directory to save annotated images.
    """

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "visualization"), exist_ok=True)

    # Define lane colors
    lane_colors = {
        "egoleft_lane": (0, 255, 0),  # Green
        "egoright_lane": (0, 0, 255),  # Blue
        "other_lanes": (255, 255, 0),  # Yellow
    }
    lane_width = 5

    number_to_store = 0

    for image_name, lanes in classified_lanes.items():
        image_path = os.path.join(gt_images_path, image_name)

        if not os.path.exists(image_path):
            print(
                f"Warning: Image {image_name} not found in {gt_images_path}. Skipping."
            )
            continue

        # Load image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw each lane
        for lane_type, lane_data in lanes.items():
            color = lane_colors.get(
                lane_type, (255, 255, 255)
            )  # Default to white if unknown type

            for lane in lane_data:
                # Use denormalized data
                lane = [(x, y) for x, y in lane]
                draw.line(lane, fill=color, width=lane_width)

        # Apply cropping if crop is provided
        if crop:
            CROP_TOP = crop["TOP"]
            CROP_RIGHT = crop["RIGHT"]
            CROP_BOTTOM = crop["BOTTOM"]
            CROP_LEFT = crop["LEFT"]

            img = img.crop(
                (
                    CROP_LEFT,
                    CROP_TOP,
                    ORIGINAL_IMG_WIDTH - CROP_RIGHT,
                    ORIGINAL_IMG_HEIGHT - CROP_BOTTOM,
                )
            )

        # Remove
        if number_to_store <= 50:
            # Save annotated image
            save_path = os.path.join(output_dir, "visualization", image_name)
            img.save(save_path)
            print(f"Saved annotated image: {save_path} with dimensions {img.size}")
            number_to_store += 1

    print("Annotation complete. All images saved in", output_dir)


def classify_lanes(normalized_data):
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


def final_data_format(data, crop=None):
    """
    Classify lanes into ego-left and ego-right based on anchor points.
    """
    if crop == None:
        raise Exception("crop values not available")

    result = {}

    if crop:
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]

    img_id_counter = 0
    for entry in data:
        image_id = str(str(img_id_counter).zfill(6))
        image_name = entry["image_name"]
        img_id_counter += 1
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
            # result[image_id] = {
            #     "egoleft_lane": [],
            #     "egoright_lane": [],
            #     "other_lanes": [],
            #     "img_height" : IMG_HEIGHT,
            #     "img_width" : IMG_WIDTH
            # }
            result[image_name] = {
                "egoleft_lane": [],
                "egoright_lane": [],
                "other_lanes": [],
                "img_height" : IMG_HEIGHT,
                "img_width" : IMG_WIDTH
            }
        else:
            left_idx, right_idx = ego_indexes
            # result[image_id] = {
            #     "egoleft_lane": [],
            #     "egoright_lane": [],
            #     "other_lanes": [],
            #     "img_height" : IMG_HEIGHT,
            #     "img_width" : IMG_WIDTH
            # }
            result[image_name] = {
                "egoleft_lane": [],
                "egoright_lane": [],
                "other_lanes": [],
                "img_height" : IMG_HEIGHT,
                "img_width" : IMG_WIDTH
            }
            dropped_lanes = 0
            for lane in entry["lanes"]:

                # crop logic
                normalized_keypoints = [
                    [(x - CROP_LEFT) / IMG_WIDTH, (y - CROP_TOP) / IMG_HEIGHT]
                    for x, y in lane["keypoints"][0]
                    if (CROP_LEFT <= x <= (ORIGINAL_IMG_WIDTH - CROP_RIGHT))
                    and (CROP_TOP <= y <= (ORIGINAL_IMG_HEIGHT - CROP_BOTTOM))
                ]
                # if lane["id"] == anchor_points[left_idx][1]:
                #     result[image_id]["egoleft_lane"].extend(normalized_keypoints)
                # elif lane["id"] == anchor_points[right_idx][1]:
                #     result[image_id]["egoright_lane"].extend(normalized_keypoints)
                # else:
                #     result[image_id]["other_lanes"].append(normalized_keypoints)

                # If image name is required. 

                if len(normalized_keypoints) < 2:
                    dropped_lanes += 1
                    continue

                if lane["id"] == anchor_points[left_idx][1]:
                    result[image_name]["egoleft_lane"].extend(normalized_keypoints)
                elif lane["id"] == anchor_points[right_idx][1]:
                    result[image_name]["egoright_lane"].extend(normalized_keypoints)
                else:
                    result[image_name]["other_lanes"].append(normalized_keypoints)

    print("Number of dropped lanes is", dropped_lanes)
    return {"root": result}


def denormalizeCoords(lane):
    """
    Normalize the coords of lane points.

    """
    return [(x * IMG_WIDTH, y * IMG_HEIGHT) for x, y in lane]


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
        # warnings.warn(
        #     f"Vertical lane detected: {lane}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2})."
        # )
        return (x1, None, None)
    # Get slope and intercept
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    # Get projected point on the image's bottom
    # TODO Handle Horizontal Lanes correctly.
    if a == 0:
        # warnings.warn(
        #     f"Horizontal lane detected: {lane}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2})."
        # )
        return (x1, None, None)
    x0 = (ORIGINAL_IMG_HEIGHT - b) / a

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


def normalize_keypoints(cropped_data):
    normalized_data = []

    # Process each image entry
    for entry in cropped_data:
        image_name = entry["name"]
        lanes = []
        for label in entry.get("labels", []):
            if "poly2d" in label:
                keypoints = [
                    [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in poly["vertices"]]
                    for poly in label["poly2d"]
                ]

                lanes.append(
                    {
                        "id": label["id"],
                        "keypoints": keypoints,
                        "laneDirection": label["attributes"]["laneDirection"],
                    }
                )

        # Store processed data
        normalized_data.append({"image_name": image_name, "lanes": lanes})
    return normalized_data


def format_data(data):
    formatted_data = []

    # Process each image entry
    for entry in data:
        image_name = entry["name"]
        lanes = []
        for label in entry.get("labels", []):
            if "poly2d" in label:
                keypoints = [
                    [[x, y] for x, y in poly["vertices"]] for poly in label["poly2d"]
                ]
                lanes.append(
                    {
                        "id": label["id"],
                        "keypoints": keypoints,
                        "laneDirection": label["attributes"]["laneDirection"],
                    }
                )

        # Store processed data
        formatted_data.append({"image_name": image_name, "lanes": lanes})
    return formatted_data


def process_binary_mask(args):

    CROP_TOP, CROP_RIGHT, CROP_BOTTOM, CROP_LEFT = args.crop

    # Check output directory exists or not
    os.makedirs(os.path.join(args.output_dir, "binary_mask"), exist_ok=True)

    # Get list of images from input directory
    image_files = [f for f in os.listdir(os.path.join(args.labels_dir, "lane", "masks", "train")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    binary_lane_mask_path = os.path.join(args.labels_dir, "lane", "masks", "train")
    print(binary_lane_mask_path)

    # Remove
    images_processed = 0

    # Process each image
    for image_name in image_files:

        # Remove 
        images_processed += 1
        if images_processed > 50 : break
        input_path = os.path.join(binary_lane_mask_path, image_name)  # Original image path
        output_path = os.path.join(args.output_dir, "binary_mask", image_name)  # Keep the same name in output directory

        # Read the binary mask image in grayscale mode
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # **Invert the binary mask** (assumes lane is black and background is white)
            img = 255 - img  # Invert colors

            # **Crop the image** (ensure dimensions are valid)
            height, width = img.shape
            cropped_img = img[CROP_TOP:height - CROP_BOTTOM, CROP_LEFT:width - CROP_RIGHT]

            # Save the processed image
            cv2.imwrite(output_path, cropped_img)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Skipped: {image_name} (Invalid image)")


if __name__ == "__main__":
    # ============================== Dataset structure ============================== #

    # Define Dimension
    ORIGINAL_IMG_WIDTH = 1280
    ORIGINAL_IMG_HEIGHT = 720

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
        type=int,
        nargs=4,
        help="Crop image: [TOP, RIGHT, BOTTOM, LEFT]. Must always be 4 ints. Non-cropped sizes are 0.",
        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
        default=[0, 140, 220, 140],
        required=False,
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

    # Update dimensions
    IMG_WIDTH = ORIGINAL_IMG_WIDTH - args.crop[1] - args.crop[3]
    IMG_HEIGHT = ORIGINAL_IMG_HEIGHT - args.crop[0] - args.crop[2]

    crop = {
        "TOP": args.crop[0],
        "RIGHT": args.crop[1],
        "BOTTOM": args.crop[2],
        "LEFT": args.crop[3],
    }

    # ============================== Format lane keypoints ============================== #

    formatted_data = format_data(data)
    # Save as new JSON file
    # output_file = os.path.join(args.output_dir, "formatted_lanes.json")

    # with open(output_file, "w") as f:
    #     json.dump(formatted_data, f, indent=4)

    # Check Data
    print(
        "formatted_data",
        json.dumps(formatted_data[:2], indent=4),
    )

    # ============================== Save binary mask ============================== #
    process_binary_mask(args)

    # ============================== Identify EgoLanes ============================== #
    classified_lanes = classify_lanes(formatted_data)
    # # Get the first three image keys
    # first_three_keys = list(classified_lanes.keys())[:3]

    # # Extract data for the first three images
    # first_three_images = {key: classified_lanes[key] for key in first_three_keys}

    # # Pretty print the result
    # print(json.dumps(first_three_images, indent=4))

    # ============================== AnnotateGT ====================================== #

    gt_images_path = os.path.join(args.image_dir, "train")
    annotateGT(classified_lanes, gt_images_path, output_dir=args.output_dir, crop=crop)

    # # Format Datapoints
    # Normalize
    # Crop

    # # ============================== Save required JSON ============================== #
    # data_copy = copy.deepcopy(data)
    # formatted_data_1 = format_data(data_copy)
    # data_json = final_data_format(formatted_data_1, crop=crop)

    # first_three_keys = list(data_json["root"].keys())[:3]
    # first_three_results = {key: data_json["root"][key] for key in first_three_keys}
    # print(json.dumps(first_three_results, indent=4))

    # # Save classified lanes as new JSON file
    # output_file = os.path.join(args.output_dir, "classified_lanes.json")

    # with open(output_file, "w") as f:
    #     json.dump(data_json, f, indent=4)

    # cropped_data = crop_data(data, args)
    # # Check Data
    # print("Cropped Data", json.dumps(cropped_data[:1], indent=4))

    # formatted_data = final_format_data(formatted_data)
    # # Check Data
    # # Assuming formatted_data is a dictionary
    # print("The length of formatted_data is", len(formatted_data["root"]))
    # # Get the first three image keys
    # first_three_keys = list(formatted_data["root"].keys())[:3]

    # # Extract the first three image results
    # first_three_results = {key: formatted_data["root"][key] for key in first_three_keys}

    # # Pretty print the results
    # print(json.dumps(first_three_results, indent=4))

    # Save classified lanes as new JSON file
    # output_file = os.path.join(args.output_dir, "classified_lanes.json")

    # with open(output_file, "w") as f:
    #     json.dump(classified_lanes, f, indent=4)
