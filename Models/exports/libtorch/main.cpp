/*
**
FILE:   main.cpp
DESC:   libTorch network input and oneDNN deployment example
**
*/

#include <assert.h>
#include <iostream>

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>

/*
**
FUNC:   main()
DESC:   example program entry point
**
*/
int main(int argc, char **argv) 
{
    // Get exported script module
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    // Create a module object to hold the network
    torch::jit::script::Module module_;
    
    // Try to load the script module
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "ERROR: Failed to load the model." << std::endl;
        return -1;
    }

    std::cerr << "INFO: Model <" << argv[1] << "> loaded successfully." << std::endl;

    // Dump the module to output to check
    std::cout << module_.dump_to_str(true, false, false) << std::endl;
  
    return 0;
}
/*
**
End of File
**
*/