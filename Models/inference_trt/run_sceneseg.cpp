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

namespace fs = std::filesystem;
using namespace nvinfer1;

// ------------------------
// Logger
// ------------------------
class TrtLogger final : public ILogger {
public:
    explicit TrtLogger(Severity s = Severity::kERROR) : severity_(s) {}
    void log(Severity s, const char* msg) noexcept override {
        if (s <= severity_) std::cerr << "[TRT] " << msg << "\n";
    }
private:
    Severity severity_;
};

// ------------------------
// Utils
// ------------------------
inline size_t volume(const Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

inline std::string dimsStr(const Dims& d) {
    std::string s = "(";
    for (int i = 0; i < d.nbDims; ++i) {
        s += std::to_string(d.d[i]);
        if (i + 1 < d.nbDims) s += ",";
    }
    s += ")";
    return s;
}

inline bool hasDynamic(const Dims& d) {
    for (int i = 0; i < d.nbDims; ++i)
        if (d.d[i] == -1) return true;
    return false;
}

std::string tensorIOModeToStr(TensorIOMode m) {
    return (m == TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
}
std::string dataTypeToStr(DataType t) {
    switch (t) {
        case DataType::kFLOAT: return "FP32";
        case DataType::kHALF:  return "FP16";
        case DataType::kINT8:  return "INT8";
        case DataType::kINT32: return "INT32";
        case DataType::kBOOL:  return "BOOL";
        case DataType::kUINT8: return "UINT8";
        default: return "UNKNOWN";
    }
}
std::string locationToStr(TensorLocation loc) {
    return (loc == TensorLocation::kDEVICE) ? "DEVICE" : "HOST";
}

void dumpEngine(ICudaEngine& engine) {
    std::cout << "=== TensorRT Engine Info (v3 tensors API) ===\n";
    std::cout << "# optimization profiles: " << engine.getNbOptimizationProfiles() << "\n";
    std::cout << "# I/O tensors: " << engine.getNbIOTensors() << "\n";
    for (int i = 0; i < engine.getNbIOTensors(); ++i) {
        const char* name = engine.getIOTensorName(i);
        auto mode = engine.getTensorIOMode(name);
        auto dt = engine.getTensorDataType(name);
        auto loc = engine.getTensorLocation(name);
        auto shape = engine.getTensorShape(name);
        std::cout << "\n[" << i << "] " << name << "\n";
        std::cout << "  kind: " << tensorIOModeToStr(mode)
                  << ", dtype: " << dataTypeToStr(dt)
                  << ", location: " << locationToStr(loc) << "\n";
        std::cout << "  engine-shape: " << dimsStr(shape) << "\n";
    }
    std::cout << "=============================================\n\n";
}

// Determine layout from shape (NCHW vs NHWC). Default NCHW if ambiguous.
enum class Layout { NCHW, NHWC };
Layout guessLayout(const Dims& d) {
    if (d.nbDims == 4) {
        if (d.d[1] == 3) return Layout::NCHW;
        if (d.d[3] == 3) return Layout::NHWC;
    }
    return Layout::NCHW;
}

// ------------------------
// RAII CUDA
// ------------------------
struct CudaDeleter { void operator()(void* p) const { if (p) cudaFree(p); } };
using CudaBuffer = std::unique_ptr<void, CudaDeleter>;
struct PinnedDeleter { void operator()(void* p) const { if (p) cudaFreeHost(p); } };
using PinnedBuffer = std::unique_ptr<void, PinnedDeleter>;

// ------------------------
// Core runner
// ------------------------
struct OutputBuf {
    std::string name;
    Dims        shape;
    DataType    dtype;
    PinnedBuffer hbuf;
    CudaBuffer   dbuf;
    size_t       numel = 0;
    size_t       nbytes = 0;
};
struct InputBuf {
    std::string  name;
    Dims         shape;     // concrete shape (after setInputShape)
    DataType     dtype;
    Layout       layout;
    PinnedBuffer hbuf;
    CudaBuffer   dbuf;
    size_t       numel = 0;
    size_t       nbytes = 0;
};

class Runner {
public:
    Runner(const std::string& enginePath, int profileIdx, bool verboseDump)
    : logger_(ILogger::Severity::kERROR), profileIdx_(profileIdx) {
        // Load engine
        std::ifstream f(enginePath, std::ios::binary);
        if (!f) throw std::runtime_error("Failed to open engine: " + enginePath);
        std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        runtime_.reset(createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");

        if (verboseDump) dumpEngine(*engine_);

        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");

        // Collect I/O names
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT) inputNames_.push_back(name);
            else outputNames_.push_back(name);
        }
        if (inputNames_.size() != 1) {
            throw std::runtime_error("This sample expects exactly 1 input tensor; found " +
                                     std::to_string(inputNames_.size()));
        }
        inputName_  = inputNames_[0];
        inputDtype_ = engine_->getTensorDataType(inputName_.c_str());
        engineShape_ = engine_->getTensorShape(inputName_.c_str());
        inputLayout_ = guessLayout(engineShape_);

        // Static vs dynamic bookkeeping
        inputIsDynamic_ = hasDynamic(engineShape_);
        if (!inputIsDynamic_) {
            if (inputLayout_ == Layout::NCHW) {
                fixedC_ = engineShape_.d[1];
                fixedH_ = engineShape_.d[2];
                fixedW_ = engineShape_.d[3];
            } else {
                fixedH_ = engineShape_.d[1];
                fixedW_ = engineShape_.d[2];
                fixedC_ = engineShape_.d[3];
            }
        }
    }

    // Prepare (or re-prepare) buffers. If static, H/W are forced to engine H/W.
    void prepareForHW(int H, int W) {
        if (!inputIsDynamic_) { H = fixedH_; W = fixedW_; }

        // Build desired input shape (N=1)
        Dims want{};
        want.nbDims = 4;
        if (inputLayout_ == Layout::NCHW) {
            int C = inputIsDynamic_ ? getChannelCountFromEngine() : fixedC_;
            want.d[0] = 1; want.d[1] = C; want.d[2] = H; want.d[3] = W;
        } else {
            int C = inputIsDynamic_ ? getChannelCountFromEngine(true) : fixedC_;
            want.d[0] = 1; want.d[1] = H; want.d[2] = W; want.d[3] = C;
        }

        // If already prepared with same shape, skip
        if (inputPrepared_ && equalDims(input_.shape, want)) return;

        // Set input shape on context (must match engine for static)
        if (!context_->setInputShape(inputName_.c_str(), want)) {
            throw std::runtime_error("setInputShape failed for " + inputName_);
        }

        // Allocate/reallocate input buffers
        input_.name   = inputName_;
        input_.dtype  = inputDtype_;
        input_.layout = inputLayout_;
        input_.shape  = context_->getTensorShape(inputName_.c_str());
        input_.numel  = volume(input_.shape);
        input_.nbytes = input_.numel * elementSize(input_.dtype);

        input_.hbuf.reset(nullptr);
        input_.dbuf.reset(nullptr);

        void* hptr = nullptr;
        if (cudaSuccess != cudaHostAlloc(&hptr, input_.nbytes, cudaHostAllocDefault))
            throw std::runtime_error("cudaHostAlloc input failed");
        input_.hbuf.reset(hptr);

        void* dptr = nullptr;
        if (cudaSuccess != cudaMalloc(&dptr, input_.nbytes))
            throw std::runtime_error("cudaMalloc input failed");
        input_.dbuf.reset(dptr);

        if (!context_->setTensorAddress(input_.name.c_str(), input_.dbuf.get()))
            throw std::runtime_error("setTensorAddress (input) failed");

        // Outputs: allocate according to concrete shapes
        outputs_.clear();
        for (const auto& oname : outputNames_) {
            Dims oshape = context_->getTensorShape(oname.c_str());
            if (hasDynamic(oshape)) {
                throw std::runtime_error("Output still dynamic after setting input shape: " + oname +
                                         " shape=" + dimsStr(oshape));
            }
            DataType odt = engine_->getTensorDataType(oname.c_str());
            size_t numel = volume(oshape);
            size_t nbytes = numel * elementSize(odt);

            OutputBuf ob;
            ob.name   = oname;
            ob.shape  = oshape;
            ob.dtype  = odt;
            ob.numel  = numel;
            ob.nbytes = nbytes;

            void* hptr2 = nullptr;
            if (cudaSuccess != cudaHostAlloc(&hptr2, nbytes, cudaHostAllocDefault))
                throw std::runtime_error("cudaHostAlloc output failed");
            ob.hbuf.reset(hptr2);

            void* dptr2 = nullptr;
            if (cudaSuccess != cudaMalloc(&dptr2, nbytes))
                throw std::runtime_error("cudaMalloc output failed");
            ob.dbuf.reset(dptr2);

            if (!context_->setTensorAddress(oname.c_str(), ob.dbuf.get()))
                throw std::runtime_error("setTensorAddress (output) failed");

            outputs_.push_back(std::move(ob));
        }

        inputPrepared_ = true;
    }

    // Preprocess a BGR frame into the pinned input host buffer
    // resizing the input ONLY IF IT IS NEEDED 
    void preprocess(const cv::Mat& bgr, int Ht, int Wt) {
        cv::Mat rgb, resized;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(Wt, Ht), 0, 0, cv::INTER_LINEAR);

        cv::Mat f32;
        resized.convertTo(f32, CV_32F, 1.0 / 255.0);
        

        //  
        if (inputLayout_ == Layout::NCHW) {
            std::vector<cv::Mat> ch;
            cv::split(f32, ch); // 3xHxW
            float* dst = reinterpret_cast<float*>(input_.hbuf.get());
            const int H = f32.rows, W = f32.cols;
            for (int c = 0; c < 3; ++c) {
                const float* src = reinterpret_cast<const float*>(ch[c].data);
                std::memcpy(dst + c * H * W, src, sizeof(float) * H * W);
            }
        } else {
            const size_t bytes = f32.total() * f32.elemSize();
            std::memcpy(input_.hbuf.get(), f32.data, bytes);
        }
    }

    // Copy H->D, enqueue, D->H, return elapsed ms
    double inferOnce() {
        cudaMemcpyAsync(input_.dbuf.get(), input_.hbuf.get(), input_.nbytes, cudaMemcpyHostToDevice, stream_);
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("enqueueV3 failed");
        }
        for (auto& ob : outputs_) {
            cudaMemcpyAsync(ob.hbuf.get(), ob.dbuf.get(), ob.nbytes, cudaMemcpyDeviceToHost, stream_);
        }
        cudaStreamSynchronize(stream_);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t1 - t0;
        return ms.count();
    }

    const std::vector<OutputBuf>& outputs() const { return outputs_; }
    const InputBuf& input() const { return input_; }
    Layout layout() const { return inputLayout_; }

    // helpers for main()
    bool isInputDynamic() const { return inputIsDynamic_; }
    int fixedH() const { return fixedH_; }
    int fixedW() const { return fixedW_; }

public: // lifecycle
    ~Runner() {
        if (stream_) cudaStreamDestroy(stream_);
    }
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;
    Runner(Runner&&) = delete;
    Runner& operator=(Runner&&) = delete;

    // Create CUDA stream and set optimization profile (async)
    void initStream() {
        if (cudaSuccess != cudaStreamCreate(&stream_))
            throw std::runtime_error("cudaStreamCreate failed");
        if (engine_->getNbOptimizationProfiles() > 1) {
            if (!context_->setOptimizationProfileAsync(profileIdx_, stream_)) {
                throw std::runtime_error("setOptimizationProfileAsync failed");
            }
        }
    }

private:
    size_t elementSize(DataType t) const {
        switch (t) {
            case DataType::kFLOAT: return 4;
            case DataType::kHALF:  return 2;
            case DataType::kINT8:  return 1;
            case DataType::kINT32: return 4;
            case DataType::kBOOL:  return 1;
            case DataType::kUINT8: return 1;
            default: return 4;
        }
    }
    bool equalDims(const Dims& a, const Dims& b) const {
        if (a.nbDims != b.nbDims) return false;
        for (int i = 0; i < a.nbDims; ++i)
            if (a.d[i] != b.d[i]) return false;
        return true;
    }
    int getChannelCountFromEngine(bool nhwc=false) const {
        Dims s = engine_->getTensorShape(inputName_.c_str());
        if (nhwc) { if (s.nbDims == 4 && s.d[3] > 0) return s.d[3]; }
        else      { if (s.nbDims == 4 && s.d[1] > 0) return s.d[1]; }
        return 3;
    }

private:
    TrtLogger logger_;
    std::unique_ptr<IRuntime> runtime_{};
    std::unique_ptr<ICudaEngine> engine_{};
    std::unique_ptr<IExecutionContext> context_{};
    cudaStream_t stream_{nullptr};

    std::vector<std::string> inputNames_, outputNames_;
    std::string inputName_;
    DataType inputDtype_{DataType::kFLOAT};
    Layout  inputLayout_{Layout::NCHW};
    Dims    engineShape_{};
    bool    inputIsDynamic_{false};
    int     fixedH_{-1}, fixedW_{-1}, fixedC_{-1};
    int     profileIdx_{0};

    bool inputPrepared_{false};
    InputBuf input_{};
    std::vector<OutputBuf> outputs_{};
};

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
