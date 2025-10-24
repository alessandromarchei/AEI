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
// RAII CUDA
// ------------------------
struct CudaDeleter { void operator()(void* p) const { if (p) cudaFree(p); } };
using CudaBuffer = std::unique_ptr<void, CudaDeleter>;
struct PinnedDeleter { void operator()(void* p) const { if (p) cudaFreeHost(p); } };
using PinnedBuffer = std::unique_ptr<void, PinnedDeleter>;

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
