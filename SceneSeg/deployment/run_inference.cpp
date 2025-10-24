#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "include/runner.hpp"
#include "include/visualization.hpp"
#include "include/utils.hpp"

namespace fs = std::filesystem;
using namespace nvinfer1;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " -e <engine.trt> -i <input_folder> "
                  << "[-r results_dir] [--suffix name] [--profile N]\n";
        return 1;
    }

    std::string enginePath, inputFolder;
    std::string resultsDir{"results/SceneSeg"};
    std::string suffix;
    int profileIdx = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-e" || a == "--engine") && i + 1 < argc) enginePath = argv[++i];
        else if ((a == "-i" || a == "--input_folder") && i + 1 < argc) inputFolder = argv[++i];
        else if ((a == "-r" || a == "--results_dir") && i + 1 < argc) resultsDir = argv[++i];
        else if ((a == "--suffix") && i + 1 < argc) suffix = argv[++i];
        else if ((a == "--profile") && i + 1 < argc) profileIdx = std::atoi(argv[++i]);
    }

    try {
        if (!fs::exists(enginePath)) throw std::runtime_error("Engine file not found: " + enginePath);
        if (!fs::exists(inputFolder)) throw std::runtime_error("Input folder not found: " + inputFolder);

        // Collect images
        std::vector<fs::path> imgs;
        for (auto& p : fs::directory_iterator(inputFolder)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png") imgs.push_back(p.path());
        }
        std::sort(imgs.begin(), imgs.end());
        if (imgs.empty()) throw std::runtime_error("No .png images found in " + inputFolder);

        // Determine dataset and scene names
        std::string sceneName = fs::path(inputFolder).filename().string();
        std::string datasetName = fs::path(inputFolder).parent_path().filename().string();
        std::string outFolder = resultsDir + "/" + datasetName + "/" + sceneName;
        if (!suffix.empty()) outFolder += "_" + suffix;
        fs::create_directories(outFolder);

        std::cout << "[INFO] Found " << imgs.size() << " images to process.\n";
        std::cout << "[INFO] Results will be saved in: " << outFolder << "\n";

        // Build runner
        Runner r(enginePath, profileIdx, /*verboseDump=*/true);
        r.initStream();

        // Decide input size
        int workH, workW;
        if (r.isInputDynamic()) {
            cv::Mat first = cv::imread(imgs.front().string(), cv::IMREAD_COLOR);
            if (first.empty()) throw std::runtime_error("Cannot read first image");
            workH = first.rows; workW = first.cols;
        } else {
            workH = r.fixedH();
            workW = r.fixedW();
        }
        r.prepareForHW(workH, workW);

        // Warmup
        cv::Mat warmupFrame = cv::imread(imgs.front().string(), cv::IMREAD_COLOR);
        if (warmupFrame.empty()) throw std::runtime_error("Cannot read warmup image");
        r.preprocess(warmupFrame, workH, workW);
        for (int i = 0; i < 50; ++i) r.inferOnce();

        // Main inference loop
        double alpha = 0.5;
        for (const auto& ip : imgs) {
            /*
                PREPROCESSING STEPS:
                - 1) Read image
                - 2) Resize to 640x320
                - 3) Normalize to [0,1] (with float32 and divide by 255)
                - 4) Reshape to CHW
            */

            //1) Read image
            cv::Mat frame = cv::imread(ip.string(), cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "[WARN] Skipping unreadable image: " << ip.filename().string() << "\n";
                continue;
            }
            
            //2) Resize image to 640x320 and 3) Normalize to [0,1] (with float32 and divide by 255) and 4) Reshape to CHW
            r.preprocess(frame, workH, workW);

            //perform inference. copy the input.hbuf -> input.dbuf, execute the network, copy output.dbuf -> output.hbuf
            double latency = r.inferOnce();

            //retrieve time in ms
            std::cout << "[INFO] Processed " << ip.filename().string()
                      << " (latency: " << latency << " ms)\n";

            if (!r.outputs().empty()) {
                //retrieve the output buffer
                const auto& ob = r.outputs().front();

                //create mask visualization
                cv::Mat blended = visualizeSegmentation(ob, frame, alpha);

                //save visualization into png file
                std::string outPath = outFolder + "/" + ip.stem().string() + ".png";
                cv::imwrite(outPath, blended);
            }
        }

        std::cout << "\n[INFO] All visualizations saved in: " << outFolder << "\n";

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 2;
    }

    return 0;
}
