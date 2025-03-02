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

def annotateGT(classified_lanes, gt_images_path, output_dir="visualization"):
    """
    Draws lane lines on images and saves them in the 'visualization' directory.

    :param classified_lanes: Dictionary containing lane keypoints.
    :param gt_images_path: Path to the directory containing ground truth images.
    :param output_dir: Directory to save annotated images.
    """

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)

    # Define lane colors
    lane_colors = {
        "egoleft_lane": (0, 255, 0),  # Green
        "egoright_lane": (0, 0, 255),  # Blue
        "other_lanes": (255, 255, 0),  # Yellow
    }
    lane_width = 5
    number_to_store = 0  # Limit image storage to 50

    img_id_counter = 0

    for image_name, lanes in classified_lanes.items():
        image_path = os.path.join(gt_images_path, image_name)
        image_id = str(str(img_id_counter).zfill(6))
        img_id_counter += 1

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
            if lane_type in ["img_height", "img_width"]:
                continue  # Skip metadata keys

            color = lane_colors.get(
                lane_type, (255, 255, 255)
            )  # Default white if unknown

            if lane_type == "other_lanes":
                # `other_lanes` contains multiple sets of keypoints, so we loop twice
                for lane_group in lane_data:
                    lane_points = [(x, y) for x, y in lane_group]  # Convert keypoints
                    if len(lane_points) > 1:  # Ensure we have enough points to draw
                        draw.line(lane_points, fill=color, width=lane_width)
            else:
                # `egoleft_lane` and `egoright_lane` contain a single set of keypoints
                for lane in lane_data:
                    lane_points = [(x, y) for x, y in lane]  # Convert keypoints
                    if len(lane_points) > 1:
                        draw.line(lane_points, fill=color, width=lane_width)
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

        # Save the first 50 images
        # TODO Save with zfill id.
        if number_to_store < 50:
            image_name_png = f"{image_id}.png"  # Convert to .png
            save_path = os.path.join(output_dir, "visualization", image_name_png)
            img.save(save_path, format="PNG")
            print(f"Saved annotated image: {save_path} with dimensions {img.size}")
            number_to_store += 1

    print("Annotation complete. All images saved in", output_dir)

def classify_lanes(data):
    """
    Classify lanes into ego-left and ego-right based on anchor points derived from poly2d vertices.
    """
    result = {}

    for entry in data:

        # Skip images with no lane data
        if "labels" not in entry or not entry["labels"]:
            continue  # Skip if no lane data is present

        image_name = entry["name"]
        anchor_points = []

        for lane in entry["labels"]:
            if "poly2d" not in lane:
                continue  # Skip if no polygonal data available

            # Extract first set of vertices (assuming poly2d is non-empty)
            vertices = lane["poly2d"][0]["vertices"]
            anchor = getLaneAnchor(vertices)

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
                "img_height": IMG_HEIGHT,
                "img_width": IMG_WIDTH,
            }
        else:
            left_idx, right_idx = ego_indexes
            result[image_name] = {
                "egoleft_lane": [],
                "egoright_lane": [],
                "other_lanes": [],
                "img_height": IMG_HEIGHT,
                "img_width": IMG_WIDTH,
            }

            for lane in entry["labels"]:
                lane_id = lane["id"]
                if lane_id == anchor_points[left_idx][1]:
                    result[image_name]["egoleft_lane"].append(
                        lane["poly2d"][0]["vertices"]
                    )
                elif lane_id == anchor_points[right_idx][1]:
                    result[image_name]["egoright_lane"].append(
                        lane["poly2d"][0]["vertices"]
                    )
                else:
                    result[image_name]["other_lanes"].append(
                        lane["poly2d"][0]["vertices"]
                    )

    return result


def format_data(data, crop):
    """
    Normalize all keypoints in the data by dividing x-coordinates by img_width
    and y-coordinates by img_height to scale them between 0 and 1.

    Args:
        data (dict): Dictionary containing image data with keypoints and dimensions

    Returns:
        dict: Dictionary with the same structure but normalized keypoints
    """
    format_data = {}

    if crop:
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]

    for image_key, image_data in data.items():

        # Create a copy of the image data
        formatted_image = image_data.copy()

        # Normalize lane keypoints
        for lane_type in ["egoleft_lane", "egoright_lane", "other_lanes"]:
            if lane_type in image_data:
                formatted_lanes = []

                for lane in image_data[lane_type]:
                    formatted_lane = []

                    for point in lane:
                        # Normalize x by width and y by height
                        if (
                            CROP_LEFT <= point[0] <= (ORIGINAL_IMG_WIDTH - CROP_RIGHT)
                        ) and (
                            CROP_TOP <= point[1] <= (ORIGINAL_IMG_HEIGHT - CROP_BOTTOM)
                        ):
                            new_x = point[0] - CROP_LEFT
                            new_y = point[1] - CROP_TOP
                            # Normalize and crop
                            formatted_lane.append(
                                [new_x / IMG_WIDTH, new_y / IMG_HEIGHT]
                            )

                    formatted_lanes.append(formatted_lane)

                # Update the list in place
                if len(formatted_lanes) >= 2:
                    formatted_image[lane_type] = formatted_lanes
                else:
                    formatted_image[lane_type] = []
                    warnings.warn(
                        f"Parsing {image_key} : Insufficient lane amount after cropping:"
                    )
                formatted_image[lane_type] = formatted_lanes

        format_data[image_key] = formatted_image

    return format_data

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
    if a == 0:
        warnings.warn(
            f"Horizontal lane detected: {lane}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2})."
        )
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

def process_binary_mask(args):

    CROP_TOP, CROP_RIGHT, CROP_BOTTOM, CROP_LEFT = args.crop

    # Check output directory exists or not
    os.makedirs(os.path.join(args.output_dir, "segmentation"), exist_ok=True)

    # Get list of images from input directory
    image_files = [
        f
        for f in os.listdir(os.path.join(args.labels_dir, "lane", "masks", "train"))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    binary_lane_mask_path = os.path.join(args.labels_dir, "lane", "masks", "train")
    # print(binary_lane_mask_path)

    img_id_counter = 0
    # Process each image
    for image_name in image_files:
        input_path = os.path.join(
            binary_lane_mask_path, image_name
        )  # Original image path
        image_id = str(str(img_id_counter).zfill(6))
        img_id_counter += 1
        output_path = os.path.join(args.output_dir, "segmentation", f"{image_id}.png")

        # Read the binary mask image in grayscale mode
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # **Invert the binary mask** (assumes lane is black and background is white)
            img = 255 - img  # Invert colors

            # Crop the image
            height, width = img.shape
            cropped_img = img[
                CROP_TOP : height - CROP_BOTTOM, CROP_LEFT : width - CROP_RIGHT
            ]

            # Save the processed image
            cv2.imwrite(output_path, cropped_img)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Skipped: {image_name} (Invalid image)")


def saveGT(gt_images_path, args):

    CROP_TOP, CROP_RIGHT, CROP_BOTTOM, CROP_LEFT = args.crop

    # Check output directory exists or not
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    # Get list of images from input directory
    image_files = [
        f
        for f in os.listdir(gt_images_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    img_id_counter = 0
    # Process each image
    for image_name in image_files:
        input_path = os.path.join(
            gt_images_path, image_name
        )  # Original image path
        image_id = str(str(img_id_counter).zfill(6))
        img_id_counter += 1
        output_path = os.path.join(args.output_dir, "images", f"{image_id}.png")

        # Read the image in RGB mode
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)  # OpenCV loads as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

        if img is not None:
            # Crop the image
            height, width, _ = img.shape
            cropped_img = img[
                CROP_TOP : height - CROP_BOTTOM, CROP_LEFT : width - CROP_RIGHT
            ]

            # Save the processed image
            cv2.imwrite(output_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))  # Convert back before saving
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Skipped: {image_name} (Invalid image)")

if __name__ == "__main__":
    # ============================== Constants ============================== #

    # Define Dimension
    ORIGINAL_IMG_WIDTH = 1280
    ORIGINAL_IMG_HEIGHT = 720

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description="Process BDD100k dataset - EgoLanes groundtruth generation"
    )

    # bdd100k/images/100k/
    parser.add_argument(
        "--image_dir",
        type=str,
        help="BDD100k ground truth image parent directory (right after extraction)",
    )

    # bdd100k/labels/
    parser.add_argument(
        "--labels_dir",
        type=str,
        help="BDD100k labels directory (right after extraction)",
        required=True,
    )

    # ./output
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

    data = os.path.join(args.labels_dir, "lane", "polygons", "lane_train.json")

    with open(data, "r") as f:
        data = json.load(f)

    print("Size of data is", len(data))

    # Check Data
    # print(json.dumps(data[:1], indent=4))

    # Save ground truth images.
    # gt_images_path = os.path.join(args.image_dir, "train")
    # saveGT(gt_images_path, args)

    # ============================== Get crop image params ============================== #

    # Updated dimensions
    IMG_WIDTH = ORIGINAL_IMG_WIDTH - args.crop[1] - args.crop[3]
    IMG_HEIGHT = ORIGINAL_IMG_HEIGHT - args.crop[0] - args.crop[2]

    crop = {
        "TOP": args.crop[0],
        "RIGHT": args.crop[1],
        "BOTTOM": args.crop[2],
        "LEFT": args.crop[3],
    }

    # ============================== Save binary mask ============================== #
    # process_binary_mask(args)

    # ============================== Identify EgoLanes ============================== #
    classified_lanes = classify_lanes(data)
    # Get the first three image keys
    first_three_keys = list(classified_lanes.keys())[:3]

    # Extract data for the first three images
    first_three_images = {key: classified_lanes[key] for key in first_three_keys}

    # Pretty print the result
    print(json.dumps(first_three_images, indent=4))

    # # ============================== AnnotateGT ====================================== #

    gt_images_path = os.path.join(args.image_dir, "train")
    annotateGT(classified_lanes, gt_images_path, output_dir=args.output_dir)

    # # # ============================== Save required JSON ============================== #
    # Normalise, crop and serialise image_names
    formatted_data = format_data(classified_lanes, crop)

    # Get the first three image keys (or all if less than three)
    # image_keys = list(formatted_data.keys())[:3]
    # # Extract data for the first three images
    # first_three_formatted_data = {key: formatted_data[key] for key in image_keys}
    # # Pretty print the result
    # print(json.dumps(first_three_formatted_data, indent=4))

    # Save classified lanes as new JSON file
    output_file = os.path.join(args.output_dir, "classified_lanes.json")

    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=4)
