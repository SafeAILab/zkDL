#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <torch/torch.h>
#include <torch/script.h>
#include <fstream>
#include <memory>

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "proof.cuh"

#include "timer.hpp"
#include "zkfc.cuh"
#include "zkrelu.cuh"

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
    torch::jit::script::Module m;
    try {
        m = torch::jit::load("traced_model.pt", device);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    for (int i = 0; i < 5; ++i) {
        // Check if the i-th module has a "weight" attribute
        if (!m.hasattr(to_string(i)) || !m.attr(to_string(i)).toModule().hasattr("weight")) {
            cout << "Layer " << i << " does not have a weight attribute." << endl;
            continue;
        }

        // Access weights of the i-th Linear layer
        auto linear_weight = m.attr(to_string(i)).toModule().attr("weight");

        // Get the weight on the GPU
        auto weight_tensor = linear_weight.toTensor().t();
        float* weight_ptr = weight_tensor.data_ptr<float>();
        cout << "Layer " << i << " weight size: " << weight_tensor.sizes() << endl;
        cout << "Layer " << i << " weight device: " << weight_tensor.device() << endl;
    }
    
	return 0;
}