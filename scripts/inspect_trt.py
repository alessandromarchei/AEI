import argparse
import os
import sys
import tensorrt as trt

DEFAULT_ENGINE_PATH = "/home/alessandro/work/autoware.privately-owned-vehicles/Models/SceneSeg_int8.trt"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def main(engine_path):
    if not os.path.isfile(engine_path):
        print(f"❌ Engine file not found: {engine_path}", file=sys.stderr)
        sys.exit(1)

    # Load engine
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Check num_bindings property or method
    if callable(engine.num_bindings):
        n_bindings = engine.num_bindings()
    else:
        n_bindings = engine.num_bindings

    print("=" * 60)
    print(f"✅ Engine loaded: {engine_path}")
    print(f"Number of bindings: {n_bindings}")
    print("=" * 60)

    # Print details for each binding
    for i in range(n_bindings):
        name = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)

        print(f"[Binding {i}]")
        print(f"  Name       : {name}")
        print(f"  Dtype      : {dtype}")
        print(f"  Shape      : {shape}")
        print(f"  Is Input   : {is_input}")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TensorRT engine bindings")
    parser.add_argument("-e", "--engine", type=str, default=DEFAULT_ENGINE_PATH,
                        help=f"path to the .trt engine file (default: {DEFAULT_ENGINE_PATH})")
    args = parser.parse_args()
    main(args.engine)
