#!/usr/bin/env python3
"""
ONNX Model Benchmarking Utility

This utility compares the performance of FP32 and INT8 quantized ONNX models.
It provides comprehensive benchmarking metrics including:
- Inference speed
- Model size
- Memory usage
- Output accuracy differences
Usage:
    python benchmark_onnx_models.py -f /path/to/fp32_model.onnx -q /path/to/int8_model.onnx -d /path/to/dataset --save_outputs
"""

import os
import time
import logging
import argparse
import numpy as np
import onnxruntime
import psutil
import platform
import cv2
from glob import glob
import random
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_model_info(model_path):
    """
    Extract metadata from an ONNX model.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        Dictionary containing model information (size, inputs, outputs, etc.)
    """
    info = {}
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
        
    # Get file size
    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    info['size_bytes'] = file_size_bytes
    info['size_mb'] = file_size_mb
    
    # Get model metadata
    try:
        session = onnxruntime.InferenceSession(model_path)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        info['input_names'] = [input.name for input in inputs]
        info['input_shapes'] = [input.shape for input in inputs]
        info['output_names'] = [output.name for output in outputs]
        
        # Extract input shape from model
        input_shape = inputs[0].shape
        # ONNX input shape is typically [batch_size, channels, height, width]
        if len(input_shape) == 4:
            info['height'] = input_shape[2] if isinstance(input_shape[2], int) else None
            info['width'] = input_shape[3] if isinstance(input_shape[3], int) else None
        
        info['providers'] = session.get_providers()
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return None
        
    return info

def load_calibration_images(calibration_data_dir, num_images=10):
    """
    Load sample images from calibration dataset for benchmarking.
    
    Args:
        calibration_data_dir: Root directory containing dataset folders
        num_images: Number of images to load for benchmarking
        
    Returns:
        List of image paths for benchmarking
    """
    # List of standard segmentation datasets
    datasets = ['ACDC', 'BDD100K', 'comma10k', 'IDDAW', 'Mapillary_Vistas', 'MUSES']
    
    all_images = []
    
    # Collect images from each dataset
    for dataset in datasets:
        # Handle nested structure: dataset/dataset/images/
        dataset_path = os.path.join(calibration_data_dir, dataset, dataset, 'images')
        if not os.path.exists(dataset_path):
            continue
            
        # Get all images from the dataset
        dataset_images = glob(os.path.join(dataset_path, "*.jpg")) + \
                       glob(os.path.join(dataset_path, "*.png"))
        
        if dataset_images:
            all_images.extend(dataset_images)
    
    if not all_images:
        logger.error(f"No images found in calibration directory: {calibration_data_dir}")
        return None
    
    # Randomly select images for benchmarking
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    logger.info(f"Selected {len(selected_images)} images for benchmarking")
    
    return selected_images

def preprocess_image(img_path, input_shape):
    """
    Preprocess an image for model inference.
    
    Processing steps:
    1. Resize to model input dimensions
    2. Convert BGR to RGB
    3. Normalize using ImageNet mean and std
    4. Convert HWC to CHW format (channels first)
    5. Add batch dimension
    
    Args:
        img_path: Path to the image file
        input_shape: Tuple of (height, width)
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            return None
            
        # Convert from BGR to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        
        # Normalize with ImageNet mean and std
        norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        img = img.astype(np.float32) / 255.0
        img = (img - norm_mean) / norm_std
        
        # HWC to CHW format (channels first for ONNX)
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Ensure float32 precision
        img = img.astype(np.float32)
        
        return img
        
    except Exception as e:
        logger.error(f"Error preprocessing image {img_path}: {str(e)}")
        return None

def save_predictions(model_name, image_path, input_tensor, output_tensor, output_dir):
    """
    Save model predictions as visualizations.
    
    Args:
        model_name: Name of the model (FP32/INT8)
        image_path: Path to the original image
        input_tensor: Input tensor used for inference
        output_tensor: Output tensor from model inference
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except Exception as e:
        logger.error(f"Error creating output directory: {str(e)}")
        return
        
    # Get original image for visualization
    try:
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            logger.error(f"Could not read image {image_path} for visualization")
            return
            
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error reading original image: {str(e)}")
        return
    
    # Get basename for saving files
    img_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the raw output as numpy array for further analysis
    try:
        np.save(os.path.join(output_dir, f"{img_basename}_{model_name}_raw.npy"), output_tensor)
    except Exception as e:
        logger.error(f"Error saving raw output: {str(e)}")
    
    try:
        # Handle different output formats based on shape
        
        # Case 1: RGB image output (shape [1, 3, H, W])
        if len(output_tensor.shape) == 4 and output_tensor.shape[1] == 3:
            # Convert from NCHW to HWC format
            output_img = output_tensor[0].transpose(1, 2, 0)
            
            # Normalize to 0-255 range if needed
            if output_img.min() < 0 or output_img.max() > 1:
                min_val = output_img.min()
                max_val = output_img.max()
                if min_val != max_val:  # Prevent division by zero
                    output_img = (output_img - min_val) / (max_val - min_val)
            
            # Convert to uint8 for saving
            output_img = (output_img * 255).astype(np.uint8)
                
            # Save the output image
            output_path = os.path.join(output_dir, f"{img_basename}_{model_name}_output.png")
            cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            
            # Create side-by-side comparison with original
            try:
                # Resize original to match output size
                orig_resized = cv2.resize(orig_img, (output_img.shape[1], output_img.shape[0]))
                
                # Create combined image
                combined = np.hstack((orig_resized, output_img))
                comparison_path = os.path.join(output_dir, f"{img_basename}_{model_name}_comparison.png")
                cv2.imwrite(comparison_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            except Exception as e:
                logger.error(f"Error creating comparison: {str(e)}")
                
        # Case 2: Segmentation output with multiple classes [1, C, H, W] where C > 3
        elif len(output_tensor.shape) == 4 and output_tensor.shape[1] > 3:
            # Remove batch dimension
            output = output_tensor[0]  # Now shape [C, H, W]
            
            # Get predicted class for each pixel (argmax across class dimension)
            predicted_mask = np.argmax(output, axis=0)
            
            # Create a colormap based on number of classes
            num_classes = output.shape[0]
            colormap = plt.cm.get_cmap('viridis', num_classes)
            
            # Convert to RGB visualization
            colored_mask = colormap(predicted_mask)
            colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
            
            # Resize mask to match original image size
            colored_mask = cv2.resize(colored_mask, (orig_img.shape[1], orig_img.shape[0]))
            
            # Create a blended visualization
            alpha = 0.5
            blended = cv2.addWeighted(orig_img, 1-alpha, colored_mask, alpha, 0)
            
            # Save all visualizations
            cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{model_name}_mask.png"), 
                        cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{model_name}_blended.png"), 
                        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            
        # Case 3: Single-channel output [1, 1, H, W]
        elif len(output_tensor.shape) == 4 and output_tensor.shape[1] == 1:
            # Extract the single channel
            heatmap = output_tensor[0, 0]
            
            # Normalize to 0-1 range if needed
            if heatmap.min() < 0 or heatmap.max() > 1:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # Create a colormap visualization
            heatmap_colored = plt.cm.jet(heatmap)
            heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
            
            # Resize to match original image
            heatmap_colored = cv2.resize(heatmap_colored, (orig_img.shape[1], orig_img.shape[0]))
            
            # Create a blended visualization
            alpha = 0.7
            blended = cv2.addWeighted(orig_img, 1-alpha, heatmap_colored, alpha, 0)
            
            # Save visualizations
            cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{model_name}_heatmap.png"), 
                       cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{model_name}_blended.png"), 
                       cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            
        # Case 4: 3D tensor without batch dimension [C, H, W]
        elif len(output_tensor.shape) == 3:
            if output_tensor.shape[0] > 1:
                # Likely segmentation with multiple classes
                predicted_mask = np.argmax(output_tensor, axis=0)
                
                # Create a colormap based on number of classes
                num_classes = output_tensor.shape[0]
                colormap = plt.cm.get_cmap('viridis', num_classes)
                
                # Convert to RGB visualization
                colored_mask = colormap(predicted_mask)
                colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
                
                # Resize mask to match original image size
                colored_mask = cv2.resize(colored_mask, (orig_img.shape[1], orig_img.shape[0]))
                
                # Create a blended visualization
                alpha = 0.5
                blended = cv2.addWeighted(orig_img, 1-alpha, colored_mask, alpha, 0)
                
                # Save all visualizations
                cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{model_name}_mask.png"), 
                            cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(output_dir, f"{img_basename}_{model_name}_blended.png"), 
                            cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                            
            else:
                # It's likely some other output format, save array visualization
                plt.figure(figsize=(10, 8))
                plt.imshow(output_tensor[0], cmap='viridis')
                plt.colorbar()
                plt.title(f"Model output - {model_name}")
                plt.savefig(os.path.join(output_dir, f"{img_basename}_{model_name}_output.png"))
                plt.close()
                
    except Exception as e:
        logger.error(f"Error during output visualization: {str(e)}")
        logger.error(traceback.format_exc())

def compare_outputs(fp32_output, int8_output, image_path, output_dir):
    """
    Compare outputs between FP32 and INT8 models and visualize differences.
    
    Theory:
        Quantization may affect model accuracy by introducing quantization error.
        This function quantifies these differences and visualizes where they occur,
        helping to assess the acceptability of accuracy loss for the application.
    
    Args:
        fp32_output: Output tensor from FP32 model
        int8_output: Output tensor from INT8 model
        image_path: Path to the original input image
        output_dir: Directory to save comparison visualizations
        
    Returns:
        Dictionary with difference statistics
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Calculate absolute difference between outputs
    abs_diff = np.abs(fp32_output - int8_output)
    
    # Calculate statistics
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    std_diff = np.std(abs_diff)
    
    # For segmentation outputs, check if class predictions differ
    if len(fp32_output.shape) == 4 and fp32_output.shape[1] > 1:  # [B, C, H, W]
        # Get class predictions
        fp32_classes = np.argmax(fp32_output[0], axis=0)
        int8_classes = np.argmax(int8_output[0], axis=0)
        
        # Count pixels where class prediction differs
        class_diff_mask = fp32_classes != int8_classes
        num_diff_pixels = np.sum(class_diff_mask)
        total_pixels = fp32_classes.size
        pct_diff = (num_diff_pixels / total_pixels) * 100
        
        # Create visualization of class differences
        diff_vis = np.zeros((*class_diff_mask.shape, 3), dtype=np.uint8)
        diff_vis[class_diff_mask] = [255, 0, 0]  # Red for differences
        
        # Save difference visualization
        cv2.imwrite(os.path.join(output_dir, f"{img_basename}_class_diffs.png"), diff_vis)
        
    # Save raw difference data
    np.save(os.path.join(output_dir, f"{img_basename}_abs_diff.npy"), abs_diff)
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff
    }

def benchmark_model(model_path, calibration_images, runs=10, warmup_runs=3, save_outputs=False, output_dir=None):
    """
    Benchmark inference speed and memory usage of an ONNX model.
    
    Theory:
        To accurately measure inference performance, this function:
        1. Performs warmup runs to ensure cache is loaded and JIT compilation is complete
        2. Uses actual calibration images to represent real-world performance
        3. Measures execution time with high-precision timer
        4. Tracks memory usage to assess resource requirements
        5. Optionally saves and visualizes model outputs
    
    Args:
        model_path: Path to the ONNX model
        calibration_images: List of paths to calibration images
        runs: Number of inference runs per image to average
        warmup_runs: Number of warmup runs before timing
        save_outputs: Whether to save model predictions
        output_dir: Directory to save output visualizations
        
    Returns:
        Dictionary with benchmark results and model outputs
    """
    results = {}
    model_outputs = {}  # Store outputs for later comparison
    logger.info(f"Benchmarking model: {model_path}")
    
    try:
        # Get model info
        model_info = get_model_info(model_path)
        if not model_info:
            return None
            
        results['model_info'] = model_info
        
        # Extract input shape from model info
        if 'height' in model_info and 'width' in model_info and model_info['height'] and model_info['width']:
            input_shape = (model_info['height'], model_info['width'])
        else:
            # Default fallback shape if not available in model
            input_shape = (320, 640)
            logger.warning(f"Could not determine input shape from model, using default: {input_shape}")
            
        logger.info(f"Using input shape: {input_shape}")
        
        # Create inference session with CPU provider for consistent benchmarking
        providers = ['CPUExecutionProvider']
        session_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            model_path, 
            sess_options=session_options,
            providers=providers
        )
        
        # Get input and output details
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        input_name = inputs[0].name
        output_names = [output.name for output in outputs]
        
        # Get model name for saving outputs
        model_type = "fp32" if "FP32" in os.path.basename(model_path).upper() else "int8"
        
        # Memory usage before session creation
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Preprocess calibration images
        processed_images = []
        for img_path in calibration_images:
            img_tensor = preprocess_image(img_path, input_shape)
            if img_tensor is not None:
                processed_images.append((img_path, img_tensor))
        
        if not processed_images:
            logger.error("No valid images could be processed for benchmarking")
            return None
            
        logger.info(f"Successfully preprocessed {len(processed_images)} images")
        
        # Warm-up runs with the first image
        logger.info(f"Performing {warmup_runs} warm-up runs...")
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: processed_images[0][1]})
        
        # Memory after warmup
        warmup_memory = process.memory_info().rss / (1024 * 1024)  # MB
        results['memory_usage_mb'] = warmup_memory - baseline_memory
        
        # Benchmark runs
        logger.info(f"Running {runs} timed inference passes per image...")
        all_run_times = []
        
        for img_idx, (img_path, img_tensor) in enumerate(processed_images):
            img_name = os.path.basename(img_path)
            logger.info(f"Image {img_idx+1}/{len(processed_images)}: {img_name}")
            
            image_run_times = []
            last_output = None
            
            for i in range(runs):
                # Clear any cached memory between runs
                if platform.system() != 'Windows':
                    os.system('sync')
                
                start = time.perf_counter()
                # Run with None for output_names to get all outputs
                outputs = session.run(None, {input_name: img_tensor})
                end = (time.perf_counter() - start) * 1000  # Convert to ms
                image_run_times.append(end)
                
                # Save the last run's output
                last_output = outputs
                
            # Log summary for this image
            avg_image_time = sum(image_run_times) / len(image_run_times)
            logger.info(f"  Average time: {avg_image_time:.2f}ms")
            all_run_times.extend(image_run_times)
            
            # Save the output for this image
            if last_output is not None:
                model_outputs[img_path] = last_output    
                # Save predictions if requested
                if save_outputs and output_dir:
                    try:
                        # Make sure the output directory exists
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # For segmentation models, the first output is typically the segmentation mask
                        save_predictions(model_type, img_path, img_tensor, last_output[0], output_dir)
                    except Exception as e:
                        logger.error(f"Error saving predictions: {str(e)}")
            else:
                logger.warning(f"No output generated for {img_name}")
            
        # Calculate statistics across all images
        avg_time = sum(all_run_times) / len(all_run_times)
        min_time = min(all_run_times)
        max_time = max(all_run_times)
        std_dev = np.std(all_run_times)
        
        results['avg_time_ms'] = avg_time
        results['min_time_ms'] = min_time
        results['max_time_ms'] = max_time
        results['std_dev_ms'] = std_dev
        results['outputs'] = model_outputs
        
        logger.info(f"Overall average inference time: {avg_time:.2f}ms (±{std_dev:.2f}ms)")
        logger.info(f"Min/Max times: {min_time:.2f}ms / {max_time:.2f}ms")
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def compare_models(fp32_results, int8_results, output_dir=None):
    """
    Compare benchmark results between FP32 and INT8 models.
    
    Theory:
        This function analyzes the trade-offs of quantization by comparing:
        1. Inference speed - The primary benefit of quantization
        2. Model size - Reduced precision means smaller models
        3. Memory usage - Memory requirements during inference
        4. Output accuracy - Potential degradation due to reduced precision
        
        The ideal outcome is significant improvements in speed and size
        with minimal impact on accuracy.
    
    Args:
        fp32_results: Results from benchmarking FP32 model
        int8_results: Results from benchmarking INT8 model
        output_dir: Directory to save comparison visualizations
    """
    if not fp32_results or not int8_results:
        logger.error("Cannot compare models: missing benchmark results")
        return
    
    logger.info("\n" + "="*50)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*50)
    
    # Performance comparison
    fp32_time = fp32_results['avg_time_ms']
    int8_time = int8_results['avg_time_ms']
    speedup = fp32_time / int8_time if int8_time > 0 else 0
    
    logger.info(f"\nPerformance:")
    logger.info(f"  FP32 average inference: {fp32_time:.2f}ms (±{fp32_results['std_dev_ms']:.2f}ms)")
    logger.info(f"  INT8 average inference: {int8_time:.2f}ms (±{int8_results['std_dev_ms']:.2f}ms)")
    logger.info(f"  Speedup: {speedup:.2f}x")
    
    # Size comparison
    fp32_size = fp32_results['model_info']['size_mb']
    int8_size = int8_results['model_info']['size_mb']
    size_reduction = fp32_size / int8_size if int8_size > 0 else 0
    
    logger.info(f"\nModel Size:")
    logger.info(f"  FP32 model: {fp32_size:.2f}MB")
    logger.info(f"  INT8 model: {int8_size:.2f}MB")
    logger.info(f"  Size reduction: {size_reduction:.2f}x")
    
    # Memory usage comparison
    fp32_memory = fp32_results['memory_usage_mb']
    int8_memory = int8_results['memory_usage_mb']
    memory_reduction = fp32_memory / int8_memory if int8_memory > 0 else 0
    
    logger.info(f"\nRuntime Memory Usage:")
    logger.info(f"  FP32 model: {fp32_memory:.2f}MB")
    logger.info(f"  INT8 model: {int8_memory:.2f}MB")
    logger.info(f"  Memory reduction: {memory_reduction:.2f}x")
    
    # Output comparison if available
    if output_dir and 'outputs' in fp32_results and 'outputs' in int8_results:
        logger.info(f"\nOutput Comparison:")
        
        # Find common images
        common_images = set(fp32_results['outputs'].keys()) & set(int8_results['outputs'].keys())
        
        if not common_images:
            logger.warning("No common images to compare outputs")
        else:
            logger.info(f"Comparing model outputs for {len(common_images)} images")
            
            all_diffs = []
            output_comp_dir = os.path.join(output_dir, "comparisons")
            if not os.path.exists(output_comp_dir):
                os.makedirs(output_comp_dir)
                
            for img_path in common_images:
                try:
                    # Get the first output tensor from each model
                    fp32_output = fp32_results['outputs'][img_path][0]
                    int8_output = int8_results['outputs'][img_path][0]
                    
                    # Ensure outputs have same shape for comparison
                    if fp32_output.shape != int8_output.shape:
                        logger.warning(f"Output shapes don't match: {fp32_output.shape} vs {int8_output.shape}")
                        continue
                    
                    diff_stats = compare_outputs(fp32_output, int8_output, img_path, output_comp_dir)
                    all_diffs.append(diff_stats)
                    
                except Exception as e:
                    logger.error(f"Error comparing outputs for {os.path.basename(img_path)}: {str(e)}")
            
            if all_diffs:
                # Calculate average differences
                avg_max_diff = sum(d['max_diff'] for d in all_diffs) / len(all_diffs)
                avg_mean_diff = sum(d['mean_diff'] for d in all_diffs) / len(all_diffs)
                
                logger.info(f"  Average maximum difference: {avg_max_diff:.6f}")
                logger.info(f"  Average mean difference: {avg_mean_diff:.6f}")
    
    logger.info("\n" + "="*50 + "\n")

def main():
    """
    Parse arguments and run benchmarks.
    """
    parser = argparse.ArgumentParser(description="Benchmark and compare ONNX models")
    parser.add_argument("-f", "--fp32_model", required=True,
                      help="Path to FP32 ONNX model")
    parser.add_argument("-q", "--int8_model", required=True,
                      help="Path to INT8 quantized ONNX model")
    parser.add_argument("-d", "--dataset", required=True,
                      help="Path to calibration dataset directory")
    parser.add_argument("--save_outputs", action="store_true",
                      help="Save model outputs and comparisons (default: False)")
    parser.add_argument("--output_dir", type=str, default="./benchmark_outputs",
                      help="Directory to save outputs (default: ./benchmark_outputs)")
    
    args = parser.parse_args()
    
    logger.info("Starting ONNX model benchmarking")
    logger.info(f"FP32 model: {args.fp32_model}")
    logger.info(f"INT8 model: {args.int8_model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info("Using default settings: 5 images, 5 runs per image, 3 warmup runs")
    
    # Set up output directory with timestamp if saving outputs
    output_dir = None
    if args.save_outputs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.output_dir}/benchmark_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Will save outputs to: {output_dir}")
        
        # Check if directory is writable
        try:
            test_file = os.path.join(output_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            logger.error(f"Output directory is not writable: {str(e)}")
            logger.error("Will continue without saving outputs")
            args.save_outputs = False
            output_dir = None

    # Get system info
    logger.info("\nSystem Information:")
    logger.info(f"  OS: {platform.system()} {platform.release()}")
    logger.info(f"  CPU: {platform.processor()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  ONNX Runtime: {onnxruntime.__version__}")
    logger.info(f"  Available providers: {onnxruntime.get_available_providers()}")
    
    # Load calibration images
    calibration_images = load_calibration_images(args.dataset)
    if not calibration_images:
        logger.error("Failed to load calibration images. Exiting.")
        return
    
    # Benchmark FP32 model
    logger.info("\nBenchmarking FP32 model...")
    fp32_results = benchmark_model(
        args.fp32_model, 
        calibration_images, 
        runs=5,
        warmup_runs=3,
        save_outputs=args.save_outputs,
        output_dir=os.path.join(output_dir, "fp32") if output_dir else None
    )
    
    # Benchmark INT8 model
    logger.info("\nBenchmarking INT8 model...")
    int8_results = benchmark_model(
        args.int8_model, 
        calibration_images, 
        runs=5,
        warmup_runs=3,
        save_outputs=args.save_outputs,
        output_dir=os.path.join(output_dir, "int8") if output_dir else None
    )
    
    # Compare results
    compare_models(fp32_results, int8_results, output_dir)

if __name__ == "__main__":
    main() 