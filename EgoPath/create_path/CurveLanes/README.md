# CurveLane Dataset Processing Script

## Overview

CurveLane is a large-scale lane detection dataset specialized in various AD/ADAS-related tasks, including lane detection, drivable path detection, object detection, path planning & decision, etc. The key feature of CurveLane, is the abundant presence of extremely curved path sections, which might take up to 40% of the 100k-image dataset. The majority of CurveLane images are in huge resolution, typically 2k and none of them are smaller than 1280x720.

## Preprocessing flow

### 1. Extra steps

CurveLane has the same vertical direction of labeling as CULane, which is lower to upper, but quite random for horizontal direction (can be any lane), which must be sorted left to right so we can reuse the CULane pipeline with ease. After that, most of the functions stay the same as CULane, except the extra processes of resizing and cropping I added in order to express the problem of the images being way too huge compared to our needs.

According to my EDA, across 100k images in CurveLanes we observe 3 different sizes:

- `2560 x 1440` : 109809 images
- `1570 x 660` : 10180 images
- `1280 x 720` : 11 images

As we agree during earlier meeting (not sure when exactly), we will convert all of em to `800 x 400`. Thus, 
the flow is as follow:

- `2560 x 1440` ==|`resize(0.5)`|==> `1280 x 720` ==|`crop(240, 160, 240, 160)`|==> `800 x 400`
- `1570 x 660` ==|`crop(130, 385, 130, 385)`|==> `800 x 400`
- `1280 x 720` ==|`crop(240, 160, 240, 160)`|==> `800 x 400`

### 2. Technical implementations

Now, most of the functions will be accompanied by 2 extra params:

- `resize (float)` : indicates resizing ratio of each image. As per the flow proposed above, our typical resizing ratio is `0.5`.
- `crop (1x4 int tuple)` : indicates crop width, being top, right, bottom, left size respectively. A crop tuple of `(a, b, c, d)` means that image will be cropped `a` pixels at top size, `b` pixels at right size, `c` pixels at bottom size, and `d` pixels at left size. Basically this order follows the one used in HTML, so I just apply it here.

Each image, upon being parsed into those functions, will have 2 params of `original_img_width/height` and `new_img_width/height`. Initially the `new` is declared the same as `original`. Each time a resize/crop action is called, the `new` is updated accordingly. This is to ensure the image's coordinates and annotations being consistently across the processing flow, since our legacy code heavily relies on the current image's size to do its biddings (this is also a pain in my ass, so I hope we could do some refactoring later, if anyone is interested).

## Usage

### Args

- `--dataset_dir` : path to CurveLane dataset directory, should contains exactly `Curvelanes` if you get it from Kaggle.
- `--output_dir` : path to directory where you wanna save the images.
- `--sampling_step` : optional. Basically tells the process to skip several images for increased model learning capability during the latter training.
- `--early_stopping` : optional. For debugging purpose. Force the process to halt upon reaching a certain amount of images. Default is 5, which means process 1 image then skip 4, and continue.

## Execute

```bash
# `pov_datasets` includes `Curvelanes` directory in this case
# Sampling rate 5 (also by default)
# Process first 100 images, then stop
python EgoPath/create_path/CurveLanes/process_curvelanes.py --dataset_dir ../pov_datasets --output_dir ../pov_datasets/Output --sampling_step 5 --early_stopping 100
```