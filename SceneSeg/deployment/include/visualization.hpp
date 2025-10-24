#pragma once

#include "utils.hpp"


inline cv::Scalar rgb2bgr(int r, int g, int b) {
    return cv::Scalar(b,g,r);
}

static cv::Mat visualizeSegmentation(const OutputBuf& ob, const cv::Mat& frame, double alpha = 0.5) {
    cv::Mat mask;

    // ---- Step 1: extract mask ----
    if (ob.dtype == DataType::kUINT8) {
        if (ob.shape.nbDims == 3) {
            int N = ob.shape.d[0], H = ob.shape.d[1], W = ob.shape.d[2];
            if (N == 1) mask = cv::Mat(H, W, CV_8UC1, const_cast<void*>(ob.hbuf.get())).clone();
        } else if (ob.shape.nbDims == 2) {
            int H = ob.shape.d[0], W = ob.shape.d[1];
            mask = cv::Mat(H, W, CV_8UC1, const_cast<void*>(ob.hbuf.get())).clone();
        }
    } else if (ob.dtype == DataType::kINT32) {
        int H = ob.shape.d[1], W = ob.shape.d[2];
        mask.create(H, W, CV_8UC1);
        const int32_t* src = reinterpret_cast<const int32_t*>(ob.hbuf.get());
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                mask.at<uint8_t>(y,x) = static_cast<uint8_t>(src[y*W+x]);
    }

    if (mask.empty()) throw std::runtime_error("Unsupported output mask shape/dtype");

    // ---- Step 2: colorize ----
    
    auto rgb2bgr = [](int r, int g, int b){ return cv::Scalar(b,g,r); };

    const cv::Scalar BG_BLUE    = rgb2bgr(61, 93, 255);
    const cv::Scalar FG_RED     = rgb2bgr(145, 28, 255);
    const cv::Scalar ROAD_GREEN = rgb2bgr(0, 255, 0);

    cv::Mat vis(mask.size(), CV_8UC3, BG_BLUE);
    vis.setTo(FG_RED,     (mask == 1));
    vis.setTo(ROAD_GREEN, (mask == 2));

    cv::Mat frame_resized;
    cv::resize(frame, frame_resized, mask.size(), 0,0, cv::INTER_LINEAR);

    cv::Mat blended;
    cv::addWeighted(vis, alpha, frame_resized, 1.0-alpha, 0.0, blended);
    return blended;
}

