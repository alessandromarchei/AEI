# deploy_libtorch
Use libTorch to load model and deploy.

## Build Instructions
export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

mkdir build && cd build

cmake -DLIBTORCH_INSTALL_ROOT=<_libTorch_root_location_> ..

make

./deploy_libtorch <_*.pt_file_> 
