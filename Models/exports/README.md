# SceneSeg Trace and Save
A script to load the SceneSeg network checkpoint file, trace and then export as a *.pt file , and an *.onnx file.

## Instructions
python3 traced_script_module_save.py -p <_network_checkpoint_file_name.pth_> -i <_test_image_file_name.png_> -o1 <_pt_output_trace_file_name.pt_> -o2 <_onnx_output_trace_file_name.onnx_>
