#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
#! /usr/bin/env python3
import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.model_components.scene_seg_network import SceneSegNetwork

from torchinfo import summary

def main(): 

    # Run inference and create visualization
    model = SceneSegNetwork()

    summary(model, input_size=(1, 3, 320, 640))



if __name__ == '__main__':
    main()
# %%