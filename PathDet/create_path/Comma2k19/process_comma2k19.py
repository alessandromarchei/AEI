import os
import sys
import numpy as np
import utils.orientation as orient
from utils.camera import img_from_device, denormalize, view_frame_from_device_frame
import cv2
import glob
import json
import argparse

# Define the region of interest (ROI)
x1, y1 = 62, 84  # Top-left corner coordinates
img_w, img_h = 1048, 524  # Width and height of the ROI

def load_frame_pos(seg_path):
    """
    Load frame positions and orientations from the given segment path.

    Parameters:
    seg_path (str): The path to the segment directory containing the global_pose data.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - frame_positions: The positions of the frames.
        - frame_orientations: The orientations of the frames.
    """
    frame_positions = np.load(seg_path + 'global_pose/frame_positions')
    frame_orientations = np.load(seg_path + 'global_pose/frame_orientations')
    return frame_positions, frame_orientations

def are_collinear(p1, p2, p3):
    """
    Check if three points are collinear.

    Parameters:
    p1 (tuple): The first point as a tuple (x, y).
    p2 (tuple): The second point as a tuple (x, y).
    p3 (tuple): The third point as a tuple (x, y).

    Returns:
    bool: True if the points are collinear, False otherwise.
    """
    # Calculate the area of the triangle formed by p1, p2, and p3
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Using the determinant method to check for collinearity
    area = 0.5 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    return area == 0

def remove_collinear_points(points):
    """
    Remove collinear points from a list of points.

    Parameters:
    points (list): A list of points as tuples (x, y).

    Returns:
    numpy.ndarray: An array of points with collinear points removed.
    """
    # Initialize the list of points to be kept
    result = [points[0]]  # Keep the first point
    
    for i in range(1, len(points) - 1):
        # Check if the middle point is collinear with the first and third points
        if not are_collinear(points[i - 1], points[i], points[i + 1]):
            result.append(points[i])
    
    result.append(points[-1])  # Always keep the last point
    
    return np.array(result)

def get_frame_positions_local(frame_count, frame_positions, frame_orientations):
    """
    Get local frame positions relative to a specific frame.

    Parameters:
    frame_count (int): The index of the frame to use as the reference.
    seg_path (str): The path to the segment directory containing the global_pose data.

    Returns:
    numpy.ndarray: An array of local frame positions.
    """
    ecef_from_local = orient.rot_from_quat(frame_orientations[frame_count])
    local_from_ecef = ecef_from_local.T
    frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions - frame_positions[frame_count])
    # print(np.linalg.norm(frame_positions[frame_count+200] - frame_positions[frame_count]))
    return frame_positions_local[frame_count+3:frame_count+200]

def extrapolate_to_bottom(points):
    """
    Extrapolate the line defined by the first two points to the bottom of the image.

    Parameters:
    points (numpy.ndarray): An array of points as tuples (x, y).

    Returns:
    numpy.ndarray: An array of points with the line extrapolated to the bottom.
    """
    x1, y1= points[0]
    x2, y2= points[1]
    
    if int(x2-x1) ==0:
        x = x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        x = int((img_h-intercept)/slope)
    return np.vstack(([x,img_h],points))

def generate_mask_vis(frame_count, seg_path, out_path, processed_img_count, 
                      frame_positions, frame_orientations,
                      width=1, height=1.2, line_color=(0,255,0)):
    """
    Generate a mask visualization for a specific frame.

    Parameters:
    frame_count (int): The index of the frame to process.
    seg_path (str): The path to the segment directory containing the images.
    width (float): The width of the path to visualize.
    height (float): The height of the path to visualize.
    fill_color (tuple): The color to fill the mask with.
    line_color (tuple): The color to draw the lines with.

    Returns:
    numpy.ndarray: An array of image points.
    """
    img = cv2.imread(out_path + f'images/{processed_img_count:05d}.png')
    device_path = get_frame_positions_local(frame_count, frame_positions, frame_orientations)
    device_path_l = device_path + np.array([0, 0, height])
    device_path_r = device_path + np.array([0, 0, height])
    device_path_l[:,1] -= width
    device_path_r[:,1] += width

    img_points_norm_l = img_from_device(device_path_l)
    img_points_norm_r = img_from_device(device_path_r)
    img_pts_l = denormalize(img_points_norm_l)
    img_pts_r = denormalize(img_points_norm_r)

    # filter out things rejected along the way
    valid = np.logical_and(np.isfinite(img_pts_l).all(axis=1), np.isfinite(img_pts_r).all(axis=1))
    img_pts_l = img_pts_l[valid].astype(int) 
    img_pts_r = img_pts_r[valid].astype(int)
    img_pts = np.array((img_pts_l+img_pts_r)/2,np.int32) -[x1,y1]
    if len(img_pts) > 190:
        img_pts = remove_collinear_points(img_pts)
        img_pts = extrapolate_to_bottom(img_pts)
        w, h = 1048, 524
        mask = np.zeros((h, w), np.uint8)
        for i in range(len(img_pts) - 1):
            cv2.line(img, img_pts[i], img_pts[i + 1], line_color, 3)
            cv2.line(mask, img_pts[i], img_pts[i + 1], (255), 3)
            
        mask_folder = out_path + "segmentation"
        vis_folder = out_path + "visualization"
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
            os.makedirs(vis_folder)
        cv2.imwrite(f"{mask_folder}/{processed_img_count:05d}.png", mask)
        cv2.imwrite(f"{vis_folder}/{processed_img_count:05d}.png", img)
        return img_pts
    else:
        os.remove(out_path + f'images/{processed_img_count:05d}.png')
        return None

def extract_frames(seg_path, out_path, img_count, downsampling_factor=1):
    """    Extract frames from a video file and save them as images.

    Parameters:
    seg_path (str): The path to the segment directory containing the video file.
    downsampling_factor (int): The factor by which to downsample the frames.

    Returns:
    int: The total number of frames extracted.
    """
    video_path = seg_path +"video.hevc"
    output_folder = out_path + "images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)

    if not vidcap.isOpened():
        print("Error opening video file")
        return

    frame_count = 0
    success, image = vidcap.read()
        
    x2, y2 = x1+img_w, y1+img_h # Bottom-right corner coordinates
    
    while success and frame_count < 1000:
        if frame_count % downsampling_factor == 0:
            img_count += 1
            cropped_image = image[y1:y2, x1:x2]
            cv2.imwrite(f"{output_folder}/{img_count:05d}.png", cropped_image)
        success, image = vidcap.read()
        frame_count += 1
    vidcap.release()
    return frame_count, img_count

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Process video segments to generate masks and visualizations.")
    parser.add_argument('--df', type=int, default=60, help='Factor by which to downsample the frames.')
    parser.add_argument('--dataset_root_path', type=str, default="", help='Root path of the dataset.')
    parser.add_argument('--out_path', type=str, default="", help='Path to the output directory.')

    args = parser.parse_args()
    downsampling_factor = args.df
    dataset_root_path = args.dataset_root_path
    out_path = args.out_path
    segments = glob.glob(dataset_root_path+"Chunk_*/*/*/")
    json_data = {"data": []}
    processed_img_count, img_count = 0, 0
    for seg in segments:
        total_frames, img_count =  extract_frames(seg, out_path, img_count, downsampling_factor)
        frame_positions, frame_orientations = load_frame_pos(seg)
        print(frame_orientations.shape, frame_positions.shape)
        for frame_count in range(0, total_frames, downsampling_factor):
            processed_img_count += 1
            drive_path = generate_mask_vis(frame_count, seg, out_path, processed_img_count,frame_positions, frame_orientations)
            if drive_path is None:
                continue
            data = {f"{img_count:05d}":{"drivable_path": drive_path.tolist(),"image_width": img_w, "image_height": img_h}}
            json_data["data"].append(data)
        print(f"Finished processing {seg}")
    with open(f"{out_path}drivable_path.json", "w") as json_file:
        json.dump(json_data, json_file)
    print("All segments processed successfully", len(segments))