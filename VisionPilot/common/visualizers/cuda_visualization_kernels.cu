#include "../include/cuda_visualization_kernels.hpp"

#ifdef CUDA_FOUND
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>

namespace autoware_pov::common {



// CUDA kernel to create masks from tensors (for backend use)
__global__ void createMaskKernel(const float* input, unsigned char* output, int rows, int cols, int channels) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= rows || y >= cols) return;

    int idx = x * cols + y;
    
    if (channels > 1) {
        // Multi-class: do argmax
        float max_score = -1e9f;
        int best_class = 0;
        
        for (int c = 0; c < channels; ++c) {
            // NCHW format: input[batch=0][channel=c][height=x][width=y]
            float score = input[c * rows * cols + x * cols + y];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }
        
        // Convert class IDs for scene segmentation: Class 1 -> 255, others -> 0
        output[idx] = (best_class == 1) ? 255 : 0;
    } else {
        // Binary: threshold
        float value = input[idx];
        output[idx] = (value > 0.0f) ? 255 : 0;
    }
}

bool CudaVisualizationKernels::createMaskFromTensorCUDA(
    const float* tensor_data,
    const std::vector<int64_t>& tensor_shape,
    cv::Mat& output_mask
) {
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount <= 0) {
        return false;
    }

    if (tensor_shape.size() != 4) {
        return false;
    }
    
    int rows = static_cast<int>(tensor_shape[2]);
    int cols = static_cast<int>(tensor_shape[3]); 
    int channels = static_cast<int>(tensor_shape[1]);
    
    size_t input_size = rows * cols * channels * sizeof(float);
    size_t output_size = rows * cols * sizeof(unsigned char);
    
    float* d_input;
    unsigned char* d_output;
    
    // Allocate GPU memory
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    
    // Copy tensor to GPU
    cudaMemcpy(d_input, tensor_data, input_size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    createMaskKernel<<<grid, block>>>(d_input, d_output, rows, cols, channels);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_input); 
        cudaFree(d_output);
        return false;
    }
    
    // Copy result back
    output_mask = cv::Mat(rows, cols, CV_8UC1);
    cudaMemcpy(output_mask.data, d_output, output_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return true;
}

// Direct tensor-to-colored visualization kernel (full pipeline)
__global__ void createColoredVisualizationKernel(
    const float* input, 
    uchar3* output, 
    int rows, 
    int cols, 
    int channels,
    int viz_type_id  // 0=scene, 1=domain
) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= rows || y >= cols) return;

    int idx = x * cols + y;
    
    uchar3 color;
    
    if (channels > 1) {
        // Multi-class: do argmax
        float max_score = -1e9f;
        int best_class = 0;
        
        for (int c = 0; c < channels; ++c) {
            float score = input[c * rows * cols + x * cols + y];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }
        
        // Color mapping based on viz type
        if (viz_type_id == 0) {  // Scene segmentation
            if (best_class == 1) {
                color = make_uchar3(0, 0, 255);  // BGR: Red for foreground
            } else {
                color = make_uchar3(0, 0, 0);    // BGR: Black for background
            }
        } else {  // Domain segmentation
            if (best_class == 0) {
                color = make_uchar3(61, 93, 255);   // BGR: Orange for road
            } else {
                color = make_uchar3(255, 28, 145);  // BGR: Purple for off-road
            }
        }
    } else {
        // Binary threshold
        float value = input[idx];
        if (value > 0.0f) {
            color = make_uchar3(0, 0, 255);  // BGR: Red
        } else {
            color = make_uchar3(0, 0, 0);    // BGR: Black
        }
    }

    output[idx] = color;
}

bool CudaVisualizationKernels::createColoredVisualizationCUDA(
    const float* tensor_data,
    const std::vector<int64_t>& tensor_shape,
    const std::string& viz_type,
    cv::Mat& colored_output
) {
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount <= 0) {
        return false;
    }

    if (tensor_shape.size() != 4) {
        return false;
    }
    
    int rows = static_cast<int>(tensor_shape[2]);
    int cols = static_cast<int>(tensor_shape[3]); 
    int channels = static_cast<int>(tensor_shape[1]);
    
    size_t input_size = rows * cols * channels * sizeof(float);
    size_t output_size = rows * cols * sizeof(uchar3);
    
    float* d_input;
    uchar3* d_output;
    
    // Allocate GPU memory
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    
    // Copy tensor to GPU
    cudaMemcpy(d_input, tensor_data, input_size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    int viz_type_id = (viz_type == "scene") ? 0 : 1;
    createColoredVisualizationKernel<<<grid, block>>>(d_input, d_output, rows, cols, channels, viz_type_id);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cudaFree(d_input); 
        cudaFree(d_output);
        return false;
    }
    
    // Copy result back
    colored_output = cv::Mat(rows, cols, CV_8UC3);
    cudaMemcpy(colored_output.data, d_output, output_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return true;
}

} // namespace autoware_pov::common

#endif // CUDA_FOUND