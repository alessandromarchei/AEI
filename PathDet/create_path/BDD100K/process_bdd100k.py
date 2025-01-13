import argparse
import json
import os
from PIL import Image, ImageDraw
import warnings
import numpy as np
import cv2

# Custom warning format
def custom_warning_format(message, category, filename, lineno, line = None):
    return f'WARNING : {message}\n'

warnings.formatwarning = custom_warning_format


# ============================== Mask Process functions ============================== #
def extractMaskFromPNG(dir):
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]

    red_min = np.array([150, 0, 0])     # Dark red lower bound
    red_max = np.array([255, 100, 100]) # Bright red upper bound

    mask_np_list = []

    for f in files:
        mask_path = os.path.join(dir, f)
        # Load the image
        img = Image.open(mask_path).convert("RGB")
        img_np = np.array(img)

        # Create a mask for red pixels
        red_mask = np.all((img_np >= red_min) & (img_np <= red_max), axis=-1).astype(np.uint8) * 255

        # Append the binary mask to the list
        mask_np_list.append(red_mask)

    # Return the list of binary masks
    return mask_np_list, files

def detectEdge(mask_np):
    """
    Detects edges in a binary mask using simple differences between adjacent pixels.
    
    :param mask_np: 2D numpy array representing the binary mask (e.g., grayscale image).
    :return: 2D numpy array with edge detection results.
    """
    # Initialize two arrays to store edge detection results
    edges1 = np.zeros_like(mask_np, dtype=np.float32)  # For forward differences
    edges2 = np.zeros_like(mask_np, dtype=np.float32)  # For backward differences

    # Iterate over each row in the mask
    for y in range(mask_np.shape[0]):
        row = mask_np[y, :]  # Extract the current row

        # Compute the forward differences between adjacent pixels
        edges1[y, :-1] = row[1:] - row[:-1]
        
        # Compute the backward differences between adjacent pixels
        edges2[y, :-1] = row[:-1] - row[1:]
    
    # Combine the absolute values of the forward and backward differences
    # Use the maximum value to capture the strongest edge response
    edges = np.maximum(np.abs(edges1), np.abs(edges2))
    
    return edges

def fromMaskToPoint(mask_np,direction = 'x'):
    """
    Change from mask point (white color) to points list.

    :param mask_np: mask in np.array format
    :param direction: x - detect row by row, y - detect column by column
    :return: List of points [(x1, y1), (x2, y2), ...]
    """
    if direction not in {'x', 'y'}:
        raise ValueError("Invalid direction! Only 'x' or 'y' are accepted.")
    
    edge_point_list = []
    if direction == 'y':
        for x in range(mask_np.shape[1]):
            column = mask_np[:, x]
            column_indices = np.where(column > 0)[0]
            for y in column_indices:
                edge_point_list.append((x,y))
    elif direction == 'x':
        for y in range(mask_np.shape[0]):
            row = mask_np[y, :]
            row_indices = np.where(row > 0)[0]
            for x in row_indices:
                edge_point_list.append((x,y))

    return edge_point_list

def fromPointToMask(point_list, img_width = 1280, img_height = 720, show = False):
    """
    Converts a list of points into a binary mask image.
    
    :param point_list: List of tuples representing points (x, y) to be drawn.
    :param img_width: Width of the output image.
    :param img_height: Height of the output image.
    :return: A numpy array representing the binary mask image.
    """
    # Create a blank image (grayscale) with initial black background
    new_image = Image.new('L', (img_width, img_height), 0)  # 'L' for grayscale, initial color 0 (black)
    
    # Initialize a drawing object
    draw_new_image = ImageDraw.Draw(new_image)
    
    # Draw each point in the point list onto the image
    for x, y in point_list:
        draw_new_image.point((x, y), fill=255)  # Set the pixel at (x, y) to white (255)
    
    if show:
        new_image.show()
    # Convert the PIL image to a numpy array for further processing
    return np.array(new_image)

def excludeTopBottomEdge(edges,x_threshold=5, y_threshold = 1):
    """
    Exclude the top and bottom of the lane
    :param edges: Edge mask in np.array format.
    :param x_threshold: Maximum allowed spacing between two points in the x-direction.
    :param y_threshold: Maximum allowed spacing between two points in the y-direction.
    :return: List of filtered edge points.
    """
    edge_point_list = fromMaskToPoint(edges,direction='y')
    
    filtered_edge_point_list = []
    for count in range(len(edge_point_list) - 1):
        next_count = count + 1

        # Extract current point
        prev_point_x = edge_point_list[count][0]
        prev_point_y = edge_point_list[count][1]

        while next_count < len(edge_point_list):
            # Extract the next point
            current_point_x = edge_point_list[next_count][0]
            current_point_y = edge_point_list[next_count][1]

            # Compute spacing in x and y directions
            x_space = np.abs(current_point_x - prev_point_x)
            y_space = np.abs(current_point_y - prev_point_y)

            # Check conditions
            if x_space != 0 and y_space <= y_threshold and x_space < x_threshold:
                filtered_edge_point_list.append((prev_point_x, prev_point_y))
                break  # Move to the next `count`

            # Increment `next_count` to evaluate the next point
            next_count += 1
    return filtered_edge_point_list

def filterOnePointEdge(edges):
    """
    Removes isolated edge points in rows of the edge mask, leaving only rows with at least two edge points.
    
    :param edges: 2D numpy array representing the edge mask.
    :return: Modified edge mask with isolated rows removed.
    """
    # Get the height (number of rows) of the edge mask
    height = edges.shape[0]
    prev_two_points = 0
    # Iterate through each row of the edge mask
    for y in range(height):
        # Extract the current row
        row = edges[y, :]
        
        # Find indices of non-zero (edge) points in the row
        edge_index = np.where(row > 0)[0]
        # print(edge_index)
        
        # If the row has fewer than two edge points, set the entire row to zero
        if len(edge_index) < 2:
            edges[y, :] = 0  # Set all elements in the row to zero (remove the row)

        interval_to_next_two_point = y - prev_two_points
        if interval_to_next_two_point >= 5:

            edges[prev_two_points, :] = 0  # Set all elements in the row to zero (remove the row)
        if len(edge_index) >= 2:
            prev_two_points = y

    # Return the modified edge mask
    return edges

def cutChippedEdge(edges, distance_criteria = 50):
    """
    Removes rows of edges that are considered 'chipped', i.e., when the distance between consecutive edges
    in the left or right side exceeds a given threshold.
    
    :param edges: 2D numpy array representing the edge mask.
    :param distance_criteria: The minimum distance between consecutive edges before a row is removed.
    :return: Modified edge mask with 'chipped' rows removed.
    """
    height = edges.shape[0]
    total_count = 0

    # Count the total number of rows with edge points
    for y in range(height):
        row = edges[y,:]
        edge_index = np.where(row>0)[0]
        if len(edge_index)>0:
            if total_count == 0:
                # Initialize previous left and right positions
                prev_left = edge_index[0]
                prev_right = edge_index[-1]
            total_count += 1

    # Counter for the rows with edge points
    count = 0

    # Iterate over the edge mask rows to detect chipped lanes
    for y in range(height):
        row = edges[y,:]
        edge_index = np.where(row>0)[0]

        if len(edge_index) > 0:
            # Calculate the distance from the previous edge positions
            left_distance = np.abs(edge_index[0] - prev_left)
            right_distance = np.abs(edge_index[-1] - prev_right)

            # Update the previous positions for the next iteration
            prev_left = edge_index[0]
            prev_right = edge_index[-1]

            # Increment the count of rows with edge points
            count += 1

            # Remove rows based on the distance criteria
            if left_distance >= distance_criteria or right_distance >= distance_criteria:
                if total_count // 2 >= count:
                    # Remove the top half of the rows if the left distance is too large
                    edges[:y,:] = 0
                else:
                    # Remove the bottom half of the rows if the right distance is too large
                    edges[y:,:] = 0
                    break

    return edges

# for debugging
def showImage(mask_np):
    image = Image.fromarray(mask_np)
    image.show()


# ============================== Helper functions ============================== #

def normalizeCoords(lane, width, height):
    """
    Normalize the coords of lane points.

    """
    return [(x / width, y / height) for x, y in lane]

def getEgoLane(edge_mask):
    """
    Identifies the leftmost and rightmost edge points for each row in an edge mask, 
    representing the ego lane boundaries.

    :param edge_mask: 2D numpy array representing the edge mask (binary image with edges).
    :return: Two lists containing the leftmost and rightmost edge points for each row:
             - left_edge_points: List of (x, y) tuples for leftmost edge points.
             - right_edge_points: List of (x, y) tuples for rightmost edge points.
    """
    # Get the dimensions of the edge mask
    height, width = edge_mask.shape

    # Initialize lists to store the leftmost and rightmost edge points
    left_edge_points = []
    right_edge_points = []

    # Iterate through each row of the edge mask
    for y in range(height):
        # Extract the current row
        row = edge_mask[y, :]

        # Find indices of non-zero (edge) points in the row
        edge_index = np.where(row > 0)[0]

        # If there are edge points in this row
        if len(edge_index) > 0:
            # Identify the leftmost edge point (first non-zero index)
            leftmost_x = edge_index[0]
            left_edge_points.append((leftmost_x, y))

            # If there are at least two edge points, identify the rightmost edge point
            if len(edge_index) >= 2:
                rightmost_x = edge_index[-1]
                right_edge_points.append((rightmost_x, y))

    # Return the lists of leftmost and rightmost edge points
    return left_edge_points, right_edge_points


def getDrivablePath(left_ego, right_ego):
    """
    Computes drivable path as midpoint between 2 ego lanes, basically the main point of this task.

    """
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

    # Extend drivable path to bottom edge of the frame
    if (len(drivable_path) >= 2):
        x1, y1 = drivable_path[-5]
        x2, y2 = drivable_path[-1]
        if (x2 == x1):
            x_bottom = x2
        else:
            a = (y2 - y1) / (x2 - x1)
            x_bottom = x2 + (img_height - y2) / a
        drivable_path.append((x_bottom, img_height))

    # Extend drivable path to be on par with longest ego lane
    # By making it parallel with longer ego lane
    y_top = min(left_ego[0][1], right_ego[0][1])
    sign_left_ego = left_ego[0][0] - left_ego[1][0]
    sign_right_ego = right_ego[0][0] - right_ego[1][0]
    sign_val = sign_left_ego * sign_right_ego
    if (sign_val > 0):  # 2 egos going the same direction
        longer_ego = left_ego if left_ego[0][1] < right_ego[0][1] else right_ego
        if len(longer_ego) >= 2 and len(drivable_path) >= 2:
            x1, y1 = longer_ego[0]
            x2, y2 = longer_ego[1]
            if (x2 == x1):
                x_top = drivable_path[0][0]
            else:
                a = (y2 - y1) / (x2 - x1)
                x_top = drivable_path[0][0] + (y_top - drivable_path[0][1]) / a

            drivable_path.insert(0, (x_top, y_top))
    else:
        # Extend drivable path to be on par with longest ego lane
        if len(drivable_path) >= 2:
            x1, y1 = drivable_path[0]
            x2, y2 = drivable_path[1]
            if (x2 == x1):
                x_top = x1
            else:
                a = (y2 - y1) / (x2 - x1)
                x_top = x1 + (y_top - y1) / a

            drivable_path.insert(0, (x_top, y_top))

    return drivable_path
    
def annotateImage(image, ego_lanes, drivable_path):
    """
    Annotate the image with detected lanes.
    - Green: Ego Lanes
    - Yellow: Drivable Path
    - Red: Other Lanes (no other lanes)
    """
    draw = ImageDraw.Draw(image)

    # Draw ego lanes in green
    for ego_lane in ego_lanes:
        draw.line([(x, y) for x, y in ego_lane], fill=(0, 255, 0), width=5)

    # Draw drivable path in yellow
    draw.line([(x, y) for x, y in drivable_path], fill=(255, 255, 0), width=5)

    return image

def drawDrivablePathMask(drivable_path):
    # Create a blank black image using PIL
    mask = Image.new('L', (img_width, img_height), 0)  # 'L' for grayscale, initial color 0 (black)
    # Draw the drivable path on the mask
    draw = ImageDraw.Draw(mask)
    draw.line(drivable_path, fill=255, width=5)  

    return mask


if __name__ == '__main__':
    # ============================== Dataset structure ============================== #

    img_width = 1280
    img_height = 720

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process BDD100k dataset - PathDet groundtruth generation"
    )

    parser.add_argument(
        "--mask_dir", 
        type = str, 
        help = "BDD100k lane mark directory (right after extraction)"
    )

    parser.add_argument(
        "--image_dir", 
        type = str, 
        help = "BDD100k image directory (right after extraction)"
    )
    parser.add_argument(
        "--output_dir", 
        type = str, 
        help = "Output directory"
    )
    args = parser.parse_args()

    # Generate output structure
    """
    --output_dir
        |----image
        |----segmentation
        |----visualization
        |----drivable_path.json
    """
    mask_dir = args.mask_dir
    image_dir = args.image_dir
    output_dir = args.output_dir

    list_subdirs = ["image", "segmentation", "visualization"]
    for subdir in list_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if (not os.path.exists(subdir_path)):
            os.makedirs(subdir_path, exist_ok = True)

    # Read the label JSON file and extract mask data and corresponding image names
    mask_np_list, name_list = extractMaskFromPNG(mask_dir)

    # Initialize the master data structure for storing processed results
    data_master = {
        'files': mask_dir,
        'data': {}
    }

    name_data = {}

    # Iterate over each mask and corresponding image
    for img_id_counter, mask_np in enumerate(mask_np_list):
        # Generate the save name for the image (5-digit format)
        save_name = str(img_id_counter).zfill(5) + ".png"

        # Get the image file path and open the image
        image_path = os.path.join(image_dir, name_list[img_id_counter].replace('png','jpg'))
        image = Image.open(image_path).convert("RGB")

        # Detect edges in the mask
        edges = detectEdge(mask_np)

        # Exclude isolated top and bottom edges from the detected edges
        new_edges_point_list = excludeTopBottomEdge(edges, x_threshold=5, y_threshold=1)

        # Convert the list of edge points back into a mask
        edges = fromPointToMask(new_edges_point_list, show=False)

        # Filter out any one-point edges from the mask
        edges = filterOnePointEdge(edges)

        # For debugging show image
        # showImage(edges)

        # Cut chipped edge from the mask
        edges = cutChippedEdge(edges)

        # For debugging show image
        # showImage(edges)

        # Detect the left and right ego lanes (representing the drivable area)
        left_ego, right_ego = getEgoLane(edge_mask=edges)

        # Generate the drivable path by connecting the left and right ego lanes
        drivable_path = getDrivablePath(left_ego, right_ego)

        # Copy the original image and save it to the 'image' directory
        copy_image = image.copy()
        copy_image.save(os.path.join(output_dir, 'image', save_name))

        # Annotate the image with the drivable path and ego lanes, then save it to the 'visualization' directory
        annotated_image = annotateImage(image.copy(), [left_ego, right_ego], drivable_path)
        annotated_image.save(os.path.join(output_dir, "visualization", save_name))

        # Generate a segmentation mask for the drivable path and save it to the 'segmentation' directory
        segmentation_mask = drawDrivablePathMask(drivable_path)
        segmentation_mask.save(os.path.join(output_dir, "segmentation", save_name))

        # Prepare annotation data for the current image (normalized drivable path coordinates)
        anno_data = {}
        anno_data[str(img_id_counter).zfill(5)] = {
            "drivable_path": normalizeCoords(drivable_path, img_width, img_height),
            "img_width": img_width,
            "img_height": img_height,
        }

        # Update the master data structure with the new annotation data
        data_master["data"].update(anno_data)

        name_data[str(img_id_counter).zfill(5)] = name_list[img_id_counter]

    # Print the total number of entries processed
    print(f"Done processing data with {len(data_master['data'])} entries in total.\n")

    # Save the final master JSON data to the output directory
    with open(os.path.join(output_dir, "drivable_path.json"), "w") as f:
        json.dump(data_master, f, indent=4)

    # Save the name JSON data to the output directory
    with open(os.path.join(output_dir, "name.json"), "w") as f:
        json.dump(name_data, f, indent=4)