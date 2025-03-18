import os
import shutil
import random
import argparse
import albumentations as A
from typing import Literal, get_args
from PIL import Image, ImageDraw
from load_data_ego_path import LoadDataEgoPath

DATA_TYPES_LITERAL = Literal['SEGMENTATION', 'DEPTH', 'KEYPOINTS']
DATA_TYPES_LIST = list(get_args(DATA_TYPES_LITERAL))

class Augmentations():
    def __init__(self, is_train: bool, data_type: DATA_TYPES_LITERAL):
        
        # Data
        self.image = 0
        self.ground_truth = 0
        self.augmented_data = 0
        self.augmented_image = 0     

        # Train vs Test/Val mode
        self.is_train = is_train

        # Dataset type
        self.data_type = data_type
        if not (self.data_type in DATA_TYPES_LIST):
            raise ValueError('Dataset type is not correctly specified')

        self.transform_shape = A.Compose(
            [
                A.Resize(width = 640, height = 320),   
                A.HorizontalFlip(p = 0.5),   
            ]
        )

        self.transform_shape_with_shuffle = A.Compose(
            [
                A.Resize(width = 640, height = 320),   
                A.HorizontalFlip(p = 0.5),
                A.RandomGridShuffle(grid=(1,2), p=0.25)   
            ]
        )

        self.transform_shape_test = A.Compose(
            [
                A.Resize(width = 640, height = 320),   
            ]
        )

        self.transform_noise = A.Compose(
            [      
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=False, p=0.5),
                A.PixelDropout(dropout_prob=0.025, per_channel=True, p=0.25),
                A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2, p=0.5),
                A.GaussNoise(var_limit=(50.0, 100.0), mean=0, noise_scale_factor=0.2, p=0.5),
                A.GaussNoise(var_limit=(250.0, 250.0), mean=0, noise_scale_factor=1, p=0.5),
                A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.5, 0.5), p=0.5),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.2, p=0.25),
                A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=1.0, alpha_coef=0.04, p=0.25),
                A.RandomRain(p=0.1),
                A.Spatter(mean=(0.65, 0.65), std=(0.3, 0.3), gauss_sigma=(2, 2), \
                    cutout_threshold=(0.68, 0.68), intensity=(0.3, 0.3), mode='rain', \
                    p=0.1),
                A.ToGray(num_output_channels=3, method='weighted_average', p=0.1)           
            ]
        )

        self.transform_shape_keypoints = A.Compose(
            [
                A.Resize(width = 640, height = 320),
                A.HorizontalFlip(p = 0.5),
                A.Rotate(limit = 10, p = 1.0)
            ]
        )

        self.transform_shape_keypoints_test = A.Compose(
            [
                A.Resize(width = 640, height = 320)
            ]
        )

    # SEMANTIC SEGMENTATION
    # Set data values
    def setDataSeg(self, image, ground_truth):

        self.image = image
        self.ground_truth = ground_truth
        
        self.augmented_data = ground_truth
        self.augmented_image = image  

    # Apply augmentations transform
    def applyTransformSeg(self, image, ground_truth):

        if(self.data_type != 'SEGMENTATION'):
            raise ValueError('Please set dataset type to SEGMENTATION in intialization of class')
        
        self.setDataSeg(image, ground_truth)

        if(self.is_train):

            # Resize and random horiztonal flip
            self.adjust_shape = self.transform_shape(image=self.image, \
                masks = self.ground_truth)
            
            self.augmented_data = self.adjust_shape["masks"]
            self.augmented_image = self.adjust_shape["image"]

            # Random image augmentations
            if (random.random() >= 0.25 and self.is_train):
        
                self.add_noise = self.transform_noise(image=self.augmented_image)
                self.augmented_image = self.add_noise["image"]
        else:
            # Only resize in test/validation mode
            self.adjust_shape = self.transform_shape_test(image=self.image, \
            masks = self.ground_truth)
            self.augmented_data = self.adjust_shape["masks"]
            self.augmented_image = self.adjust_shape["image"]

        return self.augmented_image, self.augmented_data

    # DEPTH ESTIMATION
    # Set data values
    def setDataDepth(self, image, ground_truth):

        self.image = image
        self.ground_truth = ground_truth
        self.augmented_data = ground_truth
        self.augmented_image = image  

    # Apply augmentations transform
    def applyTransformDepth(self, image, ground_truth):

        if(self.data_type != 'DEPTH'):
            raise ValueError('Please set dataset type to DEPTH in intialization of class')

        self.setDataDepth(image, ground_truth)

        if(self.is_train):

            # Resize and random horiztonal flip/grid-shuffle
            self.adjust_shape = self.transform_shape_with_shuffle(image=self.image, \
                mask=self.ground_truth)
            
            self.augmented_data = self.adjust_shape["mask"]
            self.augmented_image = self.adjust_shape["image"]

            # Random image augmentations
            if (random.random() >= 0.25 and self.is_train):
        
                self.add_noise = self.transform_noise(image=self.augmented_image)
                self.augmented_image = self.add_noise["image"]

        else:

            # Only resize in test/validation mode
            self.adjust_shape = self.transform_shape_test(image=self.image, \
                mask = self.ground_truth)
            self.augmented_data = self.adjust_shape["mask"]
            self.augmented_image = self.adjust_shape["image"]
        return self.augmented_image, self.augmented_data
    
    # KEYPOINTS
    # Set data values
    def setDataKeypoints(self, image, ground_truth):

        self.image = image
        self.augmented_image = image

        self.ground_truth = ground_truth
        self.augmented_data = ground_truth

    # Apply augmentation transform
    def applyTransformKeypoint(self, image, ground_truth):

        if (self.data_type != "KEYPOINTS"):
            raise ValueError("Please set dataset type to KEYPOINTS in intialization of class")
        
        self.setDataKeypoints(image, ground_truth)

        # For train set
        if (self.is_train):

            # Resize, random horiztonal flip and 10-deg rotate
            self.adjust_shape = self.transform_shape_keypoints(
                image = self.image,
                keypoints = self.ground_truth
            )
            
            self.augmented_image = self.adjust_shape["image"]
            self.augmented_data = self.adjust_shape["keypoints"]

            # Random image augmentations
            if (random.random() >= 0.25 and self.is_train):
        
                self.add_noise = self.transform_noise(image = self.augmented_image)
                self.augmented_image = self.add_noise["image"]

        # For test/val sets
        else:

            # Only resize in test/val
            self.adjust_shape = self.transform_shape_keypoints_test(
                image = self.image,
                keypoints = self.ground_truth
            )
            
            self.augmented_image = self.adjust_shape["image"]
            self.augmented_data = self.adjust_shape["keypoints"]

        return self.augmented_image, self.augmented_data
    

if __name__ == "__main__":

    # Testing cases here. Currently only for keypoints
    CURRENTLY_SUPPORTED_DATATYPE = [
        "KEYPOINTS"
    ]

    parser = argparse.ArgumentParser(
        description = "Testing Augmentation functions"
    )
    # Params for Augmentations
    parser.add_argument(
        "--data_type",
        type = str,
        help = "Data type, either 'SEGMENTATION', 'DEPTH', or 'KEYPOINTS'",
        required = True
    )
    # Params for LoadDataEgoPath
    parser.add_argument(
        "--image_dirpath",
        type = str,
        help = "Path to raw image directory",
        required = True
    )
    parser.add_argument(
        "--label_filepath",
        type = str,
        help = "Path to drivable_path.json",
        required = True
    )
    parser.add_argument(
        "--dataset",
        type = str,
        help = "Dataset type. Currently supporting [BDD100K, COMMA2K19, CULANE, CURVELANES, ROADWORK, TUSIMPLE]",
        required = True
    )
    parser.add_argument(
        "--set_type",
        type = Literal["train", "val"],
        help = "train or val?",
        required = True
    )
    # Output dir for the samples
    parser.add_argument(
        "--output_dir",
        type = str,
        help = "Output directory for the samples",
        required = True
    )
    # Early stopping setting
    parser.add_argument(
        "--n_samples",
        type = str,
        help = "Num. of samples you wanna limit, instead of whole image dir.",
        default = 100,
        required = False
    )
    args = parser.parse_args()

    data_type = args.data_type
    image_dirpath = args.image_dirpath
    label_filepath = args.label_filepath
    dataset = args.dataset
    set_type = args.set_type
    output_dir = args.output_dir
    n_samples = args.n_samples

    if (data_type not in CURRENTLY_SUPPORTED_DATATYPE):
        raise ValueError(f"Data type set to {data_type}. Currently supporting {CURRENTLY_SUPPORTED_DATATYPE} data types only.")
    
    if os.path.exists(output_dir):
        print(f"Output path exists. Deleting.")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Init test_data
    print(f"Initializing LoadDataEgoPath instance for dataset {dataset}...")
    test_data = LoadDataEgoPath(
        labels_filepath = label_filepath,
        images_filepath = image_dirpath,
        dataset = dataset
    )

    # Init Augmentations
    print(f"Initializing Augmentations instance...")
    aug = Augmentations(
        is_train = (set_type == "train"),
        data_type = data_type
    )

    # Extract samples from above data loader instance
    print(f"Sampling first {n_samples} samples...")
    for index in range(n_samples):

        if set_type == "train":
            np_img, ego_path = test_data.getItemTrain(index)
        elif set_type == "val":
            np_img, ego_path = test_data.getItemVal(index)
        else:
            raise ValueError(f"Cannot recognize set type {set_type}")
        
        if np_img and ego_path:
            img = Image.fromarray(np_img)
            # Renormalize ego path points
            img_width, img_height = img.size
            ego_path = [
                (point[0] * img_width, point[1] * img_height) 
                for point in ego_path
            ]
            # Augmentation
            augmented_img, augmented_ego_path = aug.applyTransformKeypoint(
                image = img,
                ground_truth = ego_path
            )
            # Draw
            lane_color = (255, 255, 0)
            lane_w = 5
            frame_name = str(index).zfill(5) + f"_{set_type}_aug.png"
            draw = ImageDraw.Draw(augmented_img)
            draw.line(
                augmented_ego_path, 
                fill = lane_color, 
                width = lane_w
            )
            # Save
            img.save(os.path.join(output_dir, frame_name))

    print(f"Sampling all done, saved at {output_dir}")