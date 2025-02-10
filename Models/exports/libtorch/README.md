# deploy_libtorch
Use libTorch to load model and deploy.

## Build Instructions
export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

mkdir build && cd build

cmake -DLIBTORCH_INSTALL_ROOT=<_libTorch_root_location_> -DOPENCV_INSTALL_ROOT=<_opencv_root_location_> ..

make

./deploy_libtorch <_input_network_file.pt_file_> <_input_image_file.png>
