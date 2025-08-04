#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoPathHead(nn.Module):
    def __init__(self):
        super(EgoPathHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.Tanh = nn.Tanh()

        # Context - MLP Layers
        self.ego_path_layer_0 = nn.Linear(800, 200)
        self.ego_path_layer_1 = nn.Linear(200, 200)
        self.ego_path_layer_2 = nn.Linear(200, 11)
 

    def forward(self, feature_vector):

        # MLP
        p0 = self.ego_path_layer_0(feature_vector)
        p0 = self.GeLU(p0)
        p1 = self.ego_path_layer_1(p0)
        p1 = self.GeLU(p1)
        p1 = self.ego_path_layer_2(p1)
        ego_path = self.Tanh(p1)*3

        return ego_path