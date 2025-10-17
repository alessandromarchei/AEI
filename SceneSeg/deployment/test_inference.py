import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically manage CUDA context
import numpy as np
import cv2
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i, binding in enumerate(engine):
        shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        size = trt.volume(shape)
        # Allocate host and device memory
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(i):
            inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
        else:
            outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    return inputs, outputs, bindings, stream

def preprocess_image(img_path, input_shape):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = input_shape[-2], input_shape[-1]
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0  # Normalize 0-1
    img = np.transpose(img, (2, 0, 1))    # HWC -> CHW
    img = np.expand_dims(img, axis=0)     # Add batch dim
    return img

def run_inference(engine_path, images_folder):
    engine = load_engine(engine_path)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    # Get input shape (assuming first input)
    input_shape = inputs[0]["shape"]
    batch_size = input_shape[0]

    # Get image paths
    img_paths = sorted([os.path.join(images_folder, f)
                        for f in os.listdir(images_folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for img_path in img_paths:
        # Preprocess
        input_img = preprocess_image(img_path, input_shape)
        np.copyto(inputs[0]["host"], input_img.ravel())

        # Transfer to GPU
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)

        # Execute
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Retrieve output
        cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
        stream.synchronize()

        output = outputs[0]["host"]
        print(f"✅ Inference done on {os.path.basename(img_path)} | Output shape: {outputs[0]['shape']}")

if __name__ == "__main__":
    engine_file = "/home/alessandro/work/autoware.privately-owned-vehicles/Models/SceneSeg_int8.trt"    # ← Change to your .engine/.trt file path
    images_dir = "/home/alessandro/work/datasets/acdc/rgb_anon_trainvaltest/rgb_anon/night/test_ref/GOPR0356"    # ← Change to your folder with images
    run_inference(engine_file, images_dir)
