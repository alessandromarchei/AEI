#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import pathlib
from PIL import Image, ImageDraw
import warnings
from datetime import datetime

# Custom warning format
def custom_warning_format(
    message, category, filename, 
    lineno, line = None
):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format


# ============================== Helper functions ============================== #


