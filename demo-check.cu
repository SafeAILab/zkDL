#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "proof.cuh"
#include <iostream>
#include <iomanip>
#include <random>
#include "timer.hpp"
#include "zkfc.cuh"
#include "zkrelu.cuh"

#include <cuda_runtime.h>
#ifdef __NVCC__
#undef __NVCC__
#include <torch/torch.h>
#include <torch/script.h>
#define __NVCC__
#else
#include <torch/torch.h>
#include <torch/script.h>
#endif

#include <fstream>
#include <memory>

using namespace std;

// FrTensor fcnn_inference(const FrTensor& X, const vector<zkFC>& fcs, vector<zkReLU>& relus, vector<FrTensor>& Z_vec, vector<FrTensor>& A_vec)
// {
//     if (fcs.size() != relus.size() + 1) throw std::runtime_error("Incompatible number of layers");
//     uint num_layer = fcs.size();
    
//     for (uint i = 0; i < num_layer - 1; ++ i)
//     {   
//         const auto& fc = fcs[i];
//         auto& relu = relus[i];
//         const auto& A = (i == 0) ? X : A_vec[i-1];

//         Z_vec.push_back(fc(A));
//         A_vec.push_back(relu(Z_vec[i]));
//     }
//     return fcs[num_layer - 1](A_vec[num_layer - 2]);
// }

ostream& operator<<(ostream& os, const FrTensor& A)
{
    cout << "[ "; 
    for (uint i = 0; i < A.size; ++ i) cout << A(i) << ' ';
    cout << ']';
    return os;
}

// const uint NUM_BITS = 16;

int main(int argc, char *argv[]) // batch_size input_dim, hidden_dim, hidden_dim, ..., output_dim
{
	torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available! Using GPU." << std::endl;
    } else {
        std::cout << "Using CPU." << std::endl;
    }


    // Load the model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("traced_model.pt", device);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    // Access weights of the first Linear layer
    auto first_linear_weight = module.attr("0").toModule().attr("weight").to(torch::kCUDA);

    // Get the first weight on the GPU
    float* weight_ptr = first_linear_weight.data_ptr<float>();

    // Print the first weight
    // std::cout << "First weight: " << weight << std::endl;

    // Timer timer;
    // timer.start();
    // cout << "Running proof on layer "<< num_layer - 1 << "..." << endl;
    // fcs[num_layer - 1].prove(A_vec[num_layer - 2], Y_hat, generators[num_layer - 1]);
    

    // for(int i = num_layer - 2; i >= 0; -- i)
    // {   
    //     cout << "Running proof on layer "<< i << "..." << endl;
    //     relus[i].prove(Z_vec[i], A_vec[i]);
    //     FrTensor& A_ = (i > 0)? A_vec[i-1] : X;
    //     fcs[i].prove(A_, Z_vec[i], generators[i]);
    // }
    // timer.stop();

    // cout << "Proof time: " << timer.getTotalTime() / batch_size << " seconds per data point." << endl;
    // cout << "Current CUDA status: " << cudaGetLastError() << endl;
	return 0;
}