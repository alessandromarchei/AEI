#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoLanesHead(nn.Module):
    def __init__(self):
        super(EgoLanesHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.dropout = nn.Dropout(p=0.5)

        # Context - MLP Layers
        self.ego_left_lane_layer_0 = nn.Linear(800, 200)
        self.ego_left_lane_layer_1 = nn.Linear(200, 200)
        self.ego_left_lane_layer_2 = nn.Linear(200, 11)

        self.ego_right_lane_layer_0 = nn.Linear(1280, 11)
        self.ego_right_lane_layer_1 = nn.Linear(200, 200)
        self.ego_right_lane_layer_2 = nn.Linear(200, 11)
 

    def forward(self, feature_vector):

        # MLP
        p0_left = self.ego_left_lane_layer_0(feature_vector)
        p0_left = self.GeLU(p0_left)
        p0_left = self.dropout(p0_left)
        p1_left = self.ego_left_lane_layer_1(p0_left)
        p1_left = self.GeLU(p1_left)
        p1_left = self.dropout(p1_left)
        ego_left_lane = self.ego_left_lane_layer_2(p1_left)

        p0_right = self.ego_right_lane_layer_0(feature_vector)
        p0_right = self.GeLU(p0_right)
        p0_right = self.dropout(p0_right)
        p1_right = self.ego_right_lane_layer_1(p0_right)
        p1_right = self.GeLU(p1_right)
        p1_right = self.dropout(p1_right)
        ego_right_lane = self.ego_right_lane_layer_2(p1_right)


        # Final result
        return ego_left_lane, ego_right_lane