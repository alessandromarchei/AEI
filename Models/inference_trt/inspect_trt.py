import tensorrt as trt

engine_file_path = "/home/alessandro/work/autoware.privately-owned-vehicles/Models/SceneSeg_int8.trt"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def main(engine_path):
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
    main(engine_file_path)
