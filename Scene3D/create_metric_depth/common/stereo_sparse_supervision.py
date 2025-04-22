#! /usr/bin/env python3
import cv2
import numpy as np

class StereoSparseSupervision():
    def __init__(self, image_left, image_right, max_height, min_height, 
                    baseline, camera_height, focal_length, cy):
        
        # SAD window size should be between 5..255
        block_size = 15

        # Matching parameters
        min_disp = 0
        num_disp = 256 - min_disp
        uniquenessRatio = 10

        # Block matcher
        stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
        stereo.setUniquenessRatio(uniquenessRatio)

        # Single channel image in OpenCV/Numpy format from PIL
        left_image_gray = np.array(image_left)[:,:,1]
        right_image_gray = np.array(image_right)[:,:,1]

        # Calculate disparity
        disparity = stereo.compute(left_image_gray, right_image_gray).astype(np.float32)/16 + 0.0001
        
        # Remove speclkes by consecutive medianBlur
        disparity = cv2.medianBlur(disparity, 5)
        disparity = cv2.medianBlur(disparity, 5)
        disparity = cv2.medianBlur(disparity, 5)
        disparity = cv2.medianBlur(disparity, 5)
        disparity = cv2.medianBlur(disparity, 5)
        disparity = cv2.medianBlur(disparity, 5)
        disparity = cv2.medianBlur(disparity, 5)
        disparity = cv2.medianBlur(disparity, 5)
        disparity = cv2.medianBlur(disparity, 5)
        disparity = cv2.medianBlur(disparity, 5)
        
        # Calculating Depth
        sparse_depth_map = baseline * focal_length / disparity
        
        # Caculating Height
        self.sparse_height_map = np.zeros_like(sparse_depth_map)
        
        size = sparse_depth_map.shape
        height = size[0]
        width = size[1]

        for i in range(0, height):
            for j in range(0, width):
                depth_val = sparse_depth_map[i, j]
                if(depth_val > 0):
                    H = (cy-i)*(depth_val)/focal_length
                    self.sparse_height_map[i,j] = H + camera_height
        
        # Clipping height values for dataset
        self.sparse_height_map = self.sparse_height_map.clip(min = min_height, max = max_height)
        

    def getSparseHeightMap(self):
        return self.sparse_height_map