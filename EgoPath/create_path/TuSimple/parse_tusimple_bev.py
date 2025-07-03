#! /usr/bin/env python3

import os
import cv2
import math
import json
import argparse
import warnings
import numpy as np
from PIL import Image, ImageDraw

# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format