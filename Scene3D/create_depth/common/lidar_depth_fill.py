#! /usr/bin/env python3
import cv2
import numpy as np


class LidarDepthFill():
    def __init__(self, depth_map):

        self.depth_map = depth_map

        # Filter kernels
        erosion_kernel = np.ones((3,3), np.uint8) 
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))

        # Morphology filters
        self.depth_map = cv2.dilate(self.depth_map, erosion_kernel, iterations=3) 
        self.depth_map = cv2.morphologyEx(self.depth_map, cv2.MORPH_CLOSE, closing_kernel)
        
        # Getting size of depth map
        size = self.depth_map.shape
        height = size[0]
        width = size[1]

        self.depth_map_fill_only = np.copy(self.depth_map)

        # Interpolate along height
        for j in range(0, width):
            interp_depth = 0
            for i in range(height-1, 1, -1):

                lower_depth = self.depth_map[i,j]
                upper_depth = self.depth_map[i-1, j]

                if(lower_depth != 0 and upper_depth == 0):
                    interp_depth = lower_depth
                
                if(interp_depth != 0 and upper_depth == 0):
                    self.depth_map[i-1, j] = interp_depth
        
        # Median blur
        self.depth_map = cv2.medianBlur(self.depth_map, 5)

    # Get filled in and interpolated lidar depth map
    def getDepthMap(self):
        return self.depth_map
    
    # Get filled in and interpolated lidar depth map
    def getDepthMapFillOnly(self):
        return self.depth_map_fill_only