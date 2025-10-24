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

#include "runner.hpp"
#include "visualization.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;
using namespace nvinfer1;



// ------------------------
// Main
// ------------------------
int main(int argc, char** argv) {

    std::cout << "=== Hello FROM THERE1 ===\n";
    std::cout << "=== Hello FROM THERE2 ===\n";
    std::cout << "=== Hello FROM THERE3 ===\n";
    std::cout << "=== Hello FROM THERE4 ===\n";
    std::cout << "=== Hello FROM THERE5 ===\n";

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " -e <engine.trt> -i <input_folder> [-o timings.csv] [-v out.avi] [--profile N]\n";
        return 1; 
    }

    std::string enginePath, inputFolder, csvOut{"inference_times.csv"}, videoOut;
    int profileIdx = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-e" || a == "--engine") && i + 1 < argc) enginePath = argv[++i];
        else if ((a == "-i" || a == "--input_folder") && i + 1 < argc) inputFolder = argv[++i];
        else if ((a == "-o" || a == "--output_csv") && i + 1 < argc) csvOut = argv[++i];
        else if ((a == "-v" || a == "--video_output_path") && i + 1 < argc) videoOut = argv[++i];
        else if (a == "--profile" && i + 1 < argc) profileIdx = std::atoi(argv[++i]);
    }

    try {
        if (!fs::exists(enginePath)) throw std::runtime_error("Engine file not found: " + enginePath);
        if (!fs::exists(inputFolder)) throw std::runtime_error("Input folder not found: " + inputFolder);

        // Collect images
        /**
         * @brief A vector containing file system paths to image files.
         *
         * This container holds the paths to images that will be processed or used
         * in the scene segmentation inference. Each element is a `fs::path` object,
         * representing the location of an image file in the file system.
         */
        std::vector<fs::path> imgs;
        for (auto& p : fs::directory_iterator(inputFolder)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png") imgs.push_back(p.path());
        }
        std::sort(imgs.begin(), imgs.end());
        if (imgs.empty()) throw std::runtime_error("No .png images found in " + inputFolder);

        // Build runner
        Runner r(enginePath, profileIdx, /*verboseDump=*/true);
        r.initStream();

        // Decide the working HxW once:
        int workH, workW;
        if (r.isInputDynamic()) {
            // dynamic: start with first frame size (can change later)
            cv::Mat first = cv::imread(imgs.front().string(), cv::IMREAD_COLOR);
            if (first.empty()) throw std::runtime_error("Cannot read first image");
            workH = first.rows; workW = first.cols;
        } else {
            // static: use engine's fixed size
            workH = r.fixedH(); workW = r.fixedW();
        }
        r.prepareForHW(workH, workW);

        // Video writer (optional)
        cv::VideoWriter writer;
        bool writeVideo = false;
        if (!videoOut.empty()) {
            cv::Mat first = cv::imread(imgs.front().string(), cv::IMREAD_COLOR);
            if (first.empty()) throw std::runtime_error("Cannot read first image for video sizing");
            int W0 = first.cols, H0 = first.rows;
            writer.open(videoOut, cv::VideoWriter::fourcc('M','J','P','G'), 5.0, cv::Size(W0, H0), true);
            if (!writer.isOpened()) throw std::runtime_error("Failed to open video writer: " + videoOut);
            writeVideo = true;
            std::cout << "[INFO] Semantic video will be saved to: " << videoOut << "\n";
        }

        // CSV
        std::ofstream csv(csvOut);
        csv << "image,inference_time_ms\n";

        // Main loop
        std::vector<double> times;
        times.reserve(imgs.size());

        //warmup cuda to run inference faster
        cv::Mat warmupFrame = cv::imread(imgs.front().string(), cv::IMREAD_COLOR);
        if (warmupFrame.empty()) {
            std::cerr << "[FATAL] Cannot read warmup image: " << imgs.front().filename().string() << "\n";
            return 1;
        }
        r.preprocess(warmupFrame, workH, workW);
        
        //warmup for 100 iterations
        for (int i = 0; i < 100; ++i) {
            r.inferOnce();
        }

        /*
        
            INFERENCE PART 
        
        */
        for (const auto& ip : imgs) {
            cv::Mat frame = cv::imread(ip.string(), cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "[WARN] Skipping unreadable image: " << ip.filename().string() << "\n";
                continue;
            }

            // If dynamic engine and frame size changed, re-prepare
            if (r.isInputDynamic()) {
                int newH = frame.rows, newW = frame.cols;
                if (newH != workH || newW != workW) {
                    r.prepareForHW(newH, newW);
                    workH = newH; workW = newW;
                }
            }

            // Preprocess into pinned buffer (always to workH x workW)
            r.preprocess(frame, workH, workW);

            // Inference
            double ms = r.inferOnce();
            times.push_back(ms);
            std::cout << "[TIMING] " << ip.filename().string() << ": " << ms << " ms\n";
            csv << ip.filename().string() << "," << ms << "\n";

            // Optional video: visualize first output
            if (writeVideo && !r.outputs().empty()) {
                const auto& ob = r.outputs().front();
                cv::Mat mask = outputToMask(ob);
                cv::Mat vis = makeVisualization(mask);
                cv::resize(vis, vis, frame.size(), 0, 0, cv::INTER_NEAREST);
                cv::Mat blended;
                cv::addWeighted(vis, 0.5, frame, 0.5, 0.0, blended);
                writer.write(blended);
            }
        }

        if (writeVideo) {
            writer.release();
            std::cout << "[INFO] Video saved to " << videoOut << "\n";
        }

        // Average
        double avg = 0.1;
        if (!times.empty()) {
            for (double t : times) avg += t;
            avg /= static_cast<double>(times.size());
        }
        std::cout << "\n[INFO] Inference times written to: " << csvOut << "\n";
        std::cout << "[AVERAGE] " << avg << " ms per image\n";

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 2;
    }

    return 0;
}
