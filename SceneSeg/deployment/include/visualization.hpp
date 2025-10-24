#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Simple mask visualization (class 0 vs others)
static cv::Mat makeVisualization(const cv::Mat& mask) {
    cv::Mat vis(mask.size(), CV_8UC3, cv::Scalar(61,93,255)); // BGR
    cv::Mat fg = (mask != 0);
    vis.setTo(cv::Scalar(255,28,145), fg);
    return vis;
}

// Try to interpret first output as [1,C,H,W] or [1,H,W] and produce a uint8 mask
static cv::Mat outputToMask(const OutputBuf& ob) {
    if (ob.dtype != DataType::kFLOAT && ob.dtype != DataType::kHALF) {
        return cv::Mat(256, 256, CV_8UC1, cv::Scalar(0)).clone();
    }
    const float* data = reinterpret_cast<const float*>(ob.hbuf.get());
    if (ob.shape.nbDims == 4) {
        int N = ob.shape.d[0], A = ob.shape.d[1], H = ob.shape.d[2], W = ob.shape.d[3];
        if (N == 1 && A > 1) {
            cv::Mat mask(H, W, CV_8UC1);
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    int best = 0; float bestv = data[0 * A * H * W + 0 * H * W + y * W + x];
                    for (int c = 1; c < A; ++c) {
                        float v = data[0 * A * H * W + c * H * W + y * W + x];
                        if (v > bestv) { bestv = v; best = c; }
                    }
                    mask.at<uint8_t>(y, x) = static_cast<uint8_t>(best);
                }
            }
            return mask;
        } else {
            cv::Mat mask(H, W, CV_8UC1);
            for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
                float v = data[y * W + x];
                mask.at<uint8_t>(y, x) = static_cast<uint8_t>(std::max(0.f, std::min(255.f, v)));
            }
            return mask;
        }
    } else if (ob.shape.nbDims == 3) {
        int N = ob.shape.d[0], H = ob.shape.d[1], W = ob.shape.d[2];
        if (N == 1) {
            cv::Mat mask(H, W, CV_8UC1);
            for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
                float v = data[y * W + x];
                mask.at<uint8_t>(y, x) = static_cast<uint8_t>(std::max(0.f, std::min(255.f, v)));
            }
            return mask;
        }
    }
    return cv::Mat(256,256,CV_8UC1, cv::Scalar(0)).clone();
}
