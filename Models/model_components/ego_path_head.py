#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoPathHead(nn.Module):
    def __init__(self):
        super(EgoPathHead, self).__init__()
        # Standard
        self.Tanh = nn.Tanh()

        # Context - MLP Layers
        self.ego_path_layer_0 = nn.Linear(800, 11)
 

    def forward(self, feature_vector):

        # MLP
        ego_path = self.ego_path_layer_0(feature_vector)
        ego_path = self.Tanh(ego_path)*3

        return ego_path