import onnx

model = onnx.load("/home/alessandro/work/autoware.privately-owned-vehicles/SceneSeg/onnx/SceneSeg_int8.onnx")
for node in model.graph.node:
    if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
        for attr in node.attribute:
            print(node.name, attr)
