#pragma once

#include "visualization.hpp"


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
    // save the resized image in input_.hbuf data structure
    void preprocess(const cv::Mat& bgr, int Ht, int Wt) {
        cv::Mat rgb, resized;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(Wt, Ht));

        cv::Mat f32;
        // Normalize to [0,1] (alfa 1.0/255.0, beta 0)
        resized.convertTo(f32, CV_32F, 1.0 / 255.0);
        

        // perform LAYOUT SEPARATION
        if (inputLayout_ == Layout::NCHW) {
            std::vector<cv::Mat> ch;
            cv::split(f32, ch); // separate the input in 3 channels

            //now copy each channel in the correct position of the input buffer (member of Runner)
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

        //input host -> device
        cudaMemcpyAsync(input_.dbuf.get(), input_.hbuf.get(), input_.nbytes, cudaMemcpyHostToDevice, stream_);
        
        //inference
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("enqueueV3 failed");
        }

        //output device -> host
        for (auto& ob : outputs_) {
            cudaMemcpyAsync(ob.hbuf.get(), ob.dbuf.get(), ob.nbytes, cudaMemcpyDeviceToHost, stream_);
        }
        //wait for all to complete. (NO DOUBLE BUFFERING HERE --> TODO)
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
