import argparse
import os
import json
import cv2

# TODO
def normalize_keypoints(lane_train_data, IMG_WIDTH, IMG_HEIGHT):
    normalized_data = []

    # Process each image entry
    for entry in lane_train_data:
        image_name = entry["name"]
        lanes = []
        for label in entry.get("labels", []):
            if "poly2d" in label:
                keypoints = [
                    [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in poly["vertices"]]
                    for poly in label["poly2d"]
                ]

                lanes.append({
                    "id": label["id"],
                    "keypoints": keypoints
                })

        # Store processed data
        normalized_data.append({
            "image_name": image_name,
            "lanes": lanes
        })
    return normalized_data


if __name__ == '__main__':
    # ============================== Dataset structure ============================== #

    # Define Dimension
    IMG_WIDTH = 1280
    IMG_HEIGHT = 720

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process BDD100k dataset - EgoLanes groundtruth generation"
    )

    # bdd100k/images/100k/train
    parser.add_argument(
        "--image_dir", 
        type = str, 
        help = "BDD100k image directory (right after extraction)",
    )

    # bdd100k/labels/
    parser.add_argument(
        "--labels_dir", 
        type = str, 
        help = "BDD100k labels directory (right after extraction)",
        required=True
    )

    parser.add_argument(
        "--output_dir", 
        type = str, 
        help = "Desired output directory",
        required=True
    )

    args = parser.parse_args()

    lane_train_json = os.path.join(args.labels_dir, "lane", "polygons", "lane_train.json")

    with open(lane_train_json, "r") as f:
        data = json.load(f)


    # Check Data
    # print(json.dumps(data[:1], indent=4)) 

    # ============================== Normalize lane keypoints ============================== #

    normalized_lane_keypoints = normalize_keypoints(data, IMG_WIDTH, IMG_HEIGHT)
    # Save as new JSON file
    output_file = os.path.join(args.output_dir, "normalized_lanes.json")

    with open(output_file, "w") as f:
        json.dump(normalize_keypoints, f, indent=4)

    # Check Data
    print("Normalized_lane_keypoints_data", json.dumps(normalized_lane_keypoints[:1], indent=4))

    # ============================== Save binary mask ============================== #
    # Check output directory exists or not
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of images from input directory
    image_files = [f for f in os.listdir(os.path.join(args.labels_dir, "lane", "masks", "train")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    binary_lane_mask_path = os.path.join(args.labels_dir, "lane", "masks", "train")
    print(binary_lane_mask_path)

    # Process each image
    for image_name in image_files:
        input_path = os.path.join(binary_lane_mask_path, image_name)  # Original image path
        output_path = os.path.join(args.output_dir, image_name)  # Keep the same name in output directory

        # Read and write the image
        img = cv2.imread(input_path)
        if img is not None:
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")
        else:
            print(f"Skipped: {image_name} (Not a valid image)")

    
    

    

    



