#pragma once
#include "structs.hpp"

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

