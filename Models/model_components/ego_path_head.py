#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoPathHead(nn.Module):
    def __init__(self):
        super(EgoPathHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.dropout = nn.Dropout(p=0.25)
        self.dropout_aggressize = nn.Dropout(p=0.4)
        self.TanH = nn.Tanh()

        # Ego Path  Decode layers
        self.ego_path_layer_0 = nn.Linear(1456, 1280)
        self.ego_path_layer_1 = nn.Linear(1280, 1024)
        self.ego_path_layer_2 = nn.Linear(1024, 800)
        self.ego_path_layer_3 = nn.Linear(800, 11)
 

    def forward(self, feature_vector):

        # Features
        feature_vector = self.dropout(feature_vector)

        # Ego Path Lane MLP
        ego_path = self.ego_path_layer_0(feature_vector)
        ego_path = self.dropout_aggressize(ego_path)
        ego_path = self.GeLU(ego_path)

        ego_path = self.ego_path_layer_1(ego_path)
        ego_path = self.dropout_aggressize(ego_path)
        ego_path = self.GeLU(ego_path)

        ego_path = self.ego_path_layer_2(ego_path)
        ego_path = self.dropout_aggressize(ego_path)
        ego_path = self.GeLU(ego_path)

        ego_path = self.ego_path_layer_3(ego_path)

        # Final Path Prediction
        ego_path = self.TanH(ego_path)*3 + 0.5

        return ego_path