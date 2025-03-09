#! /usr/bin/env python3
import torch.nn as nn

class Scene3DHead(nn.Module):
    def __init__(self):
        super(Scene3DHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()

        # Segmentation Head - Output Layers
        self.upsample_layer_3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_3 = nn.Conv2d(32, 256, 1)
        self.decode_layer_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_7 = nn.Conv2d(256, 128, 3, 1, 1)
        self.upsample_layer_4 = nn.ConvTranspose2d(128, 128, 2, 2)

        # Prediction 1
        self.decode_layer_8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_9 = nn.Conv2d(128, 64, 3, 1, 1)
        self.decode_layer_10 = nn.Conv2d(64, 1, 3, 1, 1)

        # Prediction 2
        self.decode_layer_11 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_12 = nn.Conv2d(128, 64, 3, 1, 1)
        self.decode_layer_13 = nn.Conv2d(64, 1, 3, 1, 1)

        # Prediction 3
        self.decode_layer_14 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_15 = nn.Conv2d(128, 64, 3, 1, 1)
        self.decode_layer_16 = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, neck, features):

        # Decoder upsample block 4
        # Upsample
        d7 = self.upsample_layer_3(neck)
         # Expand and add layer from Encoder
        d7 = d7 + self.skip_link_layer_3(features[0])
        # Double convolution
        d7 = self.decode_layer_6(d7)
        d7 = self.GeLU(d7)
        d8 = self.decode_layer_7(d7)
        d8 = self.GeLU(d8)

        # Decoder upsample block 5
        # Upsample
        up = self.upsample_layer_4(d8)

        # Triple convolution - output block 1
        d8_out = self.decode_layer_8(up)
        d8_out = self.GeLU(d8_out)
        d9_out = self.decode_layer_9(d8_out)
        d9_out = self.GeLU(d9_out)
        pred_1 = self.decode_layer_10(d9_out)

        # Triple convolution - output block 2
        d11_out = self.decode_layer_11(up)
        d11_out = self.GeLU(d11_out)
        d12_out = self.decode_layer_12(d11_out)
        d12_out = self.GeLU(d12_out)
        pred_2 = self.decode_layer_13(d12_out)

        # Triple convolution - output block 2
        d14_out = self.decode_layer_14(up)
        d14_out = self.GeLU(d14_out)
        d15_out = self.decode_layer_15(d14_out)
        d15_out = self.GeLU(d15_out)
        pred_3 = self.decode_layer_16(d15_out)

        # Final prediction
        prediction = pred_1 + pred_2 + pred_3
        return prediction, pred_1, pred_2, pred_3