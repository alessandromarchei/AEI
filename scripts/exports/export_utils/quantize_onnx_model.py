#!/usr/bin/env python3
"""
ONNX Model Quantization Utility

This script converts a floating-point (FP32) ONNX model to a quantized INT8 version,
which reduces model size and improves inference speed while maintaining acceptable accuracy.

Theory:
- Quantization maps floating-point values to integers (typically INT8) using scaling factors
- Post-training static quantization uses representative data (calibration dataset) to determine optimal scaling factors
- QDQ format inserts QuantizeLinear/DeQuantizeLinear nodes in the computation graph
"""
import os
import logging
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationMethod
import cv2
from argparse import ArgumentParser
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SceneSegCalibrationDataReader(CalibrationDataReader):
    """
    Prepares calibration data for quantizing scene segmentation models.
    
    Calibration requires representative input data to determine optimal quantization parameters.
    This class loads and preprocesses images from multiple datasets to ensure diverse calibration.
    
    Args:
        calibration_image_dir: Root directory containing dataset folders
        input_shape: Model input dimensions (height, width)
        images_per_dataset: Number of images to sample from each dataset
    """
    def __init__(self, calibration_image_dir, input_shape=(320, 640), images_per_dataset=200):
        self.image_dir = calibration_image_dir
        self.input_shape = input_shape
        self.images_per_dataset = images_per_dataset
        
        # Standard datasets for scene segmentation calibration
        #self.datasets = ['ACDC', 'BDD100K', 'comma10k', 'IDDAW', 'Mapillary_Vistas', 'MUSES']
        self.datasets = ['fog', 'rain', 'snow', 'night']
        
        # Collect and prepare calibration images
        self.image_list = self._collect_calibration_images()
        
        # Normalization parameters (ImageNet mean and std)
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        self.cur_index = 0
        
        if len(self.image_list) == 0:
            raise ValueError("No calibration images found! Please check the dataset paths.")

    # def _collect_calibration_images(self):
    #     """Collects calibration images from all datasets."""
    #     image_list = []
    #     logger.info("Starting calibration dataset collection...")
        
    #     for dataset in self.datasets:
    #         # Handle nested structure: dataset/dataset/images/
    #         dataset_path = os.path.join(self.image_dir, dataset)
    #         if not os.path.exists(dataset_path):
    #             logger.warning(f"Dataset images not found in {dataset_path}")
    #             continue
                
    #         # Get all images from the dataset
    #         dataset_images = glob(os.path.join(dataset_path, "*.jpg")) + \
    #                        glob(os.path.join(dataset_path, "*.png"))
            
    #         if len(dataset_images) > 0:
    #             # Sample images from this dataset
    #             selected_images = np.random.choice(
    #                 dataset_images,
    #                 size=min(self.images_per_dataset, len(dataset_images)),
    #                 replace=False
    #             ).tolist()
    #             image_list.extend(selected_images)
    #             logger.info(f"Added {len(selected_images)} images from {dataset}")
    #         else:
    #             logger.warning(f"No images found in {dataset_path}")
        
    #     # Shuffle the combined image list
    #     np.random.shuffle(image_list)
    #     logger.info(f"Total calibration images selected: {len(image_list)}")
        
    #     return image_list

    def _collect_calibration_images(self):
        """Recursively collects calibration images from all `*/train/*/*` folders inside the given datasets."""
        image_list = []
        logger.info("Starting calibration dataset collection...")

        for dataset in self.datasets:
            dataset_path = os.path.join(self.image_dir, dataset, "train")

            if not os.path.exists(dataset_path):
                logger.warning(f"Missing dataset train path: {dataset_path}")
                continue

            # Recursively search for images inside all subfolders of the train split
            dataset_images = glob(os.path.join(dataset_path, "*", "*.[jp][pn]g"), recursive=True)

            if len(dataset_images) > 0:
                selected_images = np.random.choice(
                    dataset_images,
                    size=min(self.images_per_dataset, len(dataset_images)),
                    replace=False
                ).tolist()
                image_list.extend(selected_images)
                logger.info(f"Added {len(selected_images)} images from {dataset}")
            else:
                logger.warning(f"No images found in {dataset_path}")

        np.random.shuffle(image_list)
        logger.info(f"Total calibration images selected: {len(image_list)}")
        return image_list

    def get_next(self):
        """
        Returns the next preprocessed calibration image with detailed logging.
        """
        import time
        t0 = time.time()

        if self.cur_index >= len(self.image_list):
            return None

        img_path = self.image_list[self.cur_index]
        self.cur_index += 1

        
        logger.info(f"[{self.cur_index}/{len(self.image_list)}] Loading image: {img_path}")

        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                return self.get_next()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

            img = img.astype(np.float32) / 255.0
            img = (img - self.norm_mean) / self.norm_std
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0).astype(np.float32)

            t1 = time.time()
            logger.info(f"[{self.cur_index}/{len(self.image_list)}] Processed in {t1 - t0:.3f}s")

            return {'input': img}

        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")
            return self.get_next()


def main():
    """
    Main function to run the quantization process.
    
    Handles command-line arguments, sets up the calibration data reader,
    and performs the actual quantization.
    """
    # Parse command line arguments
    parser = ArgumentParser(description="Quantize ONNX models from FP32 to INT8")
    parser.add_argument("-i", "--input_model", required=True,
                      help="Path to input FP32 ONNX model")
    parser.add_argument("-o", "--output_model", required=True,
                      help="Path to save quantized INT8 model")
    parser.add_argument("-c", "--calibration_data_dir", required=True,
                      help="Path to directory containing calibration images")
    parser.add_argument("--images_per_dataset", type=int, default=200,
                      help="Number of images to sample from each dataset (default: 200)")
    
    args = parser.parse_args()

    logger.info(f"Starting quantization: {args.input_model} → {args.output_model}")
    logger.info(f"Calibration data: {args.calibration_data_dir}")
    logger.info(f"Using {args.images_per_dataset} images per dataset for calibration")

    try:
        # Initialize calibration data reader
        calibration_data_reader = SceneSegCalibrationDataReader(args.calibration_data_dir, images_per_dataset=args.images_per_dataset)
        
        # Configure quantization parameters
        # QDQ format uses QuantizeLinear/DeQuantizeLinear nodes for optimal compatibility
        # MinMax calibration finds scaling factors using min/max values in calibration data
        logger.info("Performing INT8 quantization with QDQ format and MinMax calibration")
        quantize_static(
            model_input=args.input_model,
            model_output=args.output_model,
            calibration_data_reader=calibration_data_reader,
            # Quantization configuration
            quant_format=QuantFormat.QDQ,  # QDQ format for better compatibility
            per_channel=True,              # Per-channel quantization for weights
            reduce_range=False,             # Use 7-bit range for activations on some targets
            activation_type=QuantType.QInt8,  # 8-bit integer activations
            weight_type=QuantType.QInt8,      # 8-bit integer weights
            calibrate_method=CalibrationMethod.MinMax,  # Use min/max values for scaling
            # Advanced quantization options
            extra_options={
                'ActivationSymmetric': True,     # Use symmetric quantization for activations
                'WeightSymmetric': True,         # Use symmetric quantization for weights
                'EnableSubgraph': True,          # Quantize subgraphs
                'ForceQuantizeNoInputCheck': True,  # Force quantize specified nodes
                'CalibMovingAverage': True,         # Use moving average for calibration

            }
        )
        logger.info(f"Quantization complete: {args.output_model}")

        # Verify the quantized model
        model = onnx.load(args.output_model)
        onnx.checker.check_model(model)
        logger.info("Quantized model verification passed")

    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()