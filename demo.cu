#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <torch/torch.h>
#include <torch/script.h>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "proof.cuh"

#include "timer.hpp"
#include "zkfc.cuh"
#include "zkrelu.cuh"

using namespace std;

FrTensor fcnn_inference(const FrTensor& X, const vector<zkFC>& fcs, vector<zkReLU>& relus, vector<FrTensor>& Z_vec, vector<FrTensor>& A_vec)
{
    if (fcs.size() != relus.size() + 1) throw std::runtime_error("Incompatible number of layers");
    uint num_layer = fcs.size();
    
    for (uint i = 0; i < num_layer - 1; ++ i)
    {   
        const auto& fc = fcs[i];
        auto& relu = relus[i];
        const auto& A = (i == 0) ? X : A_vec[i-1];

        Z_vec.push_back(fc(A));
        A_vec.push_back(relu(Z_vec[i]));
    }
    return fcs[num_layer - 1](A_vec[num_layer - 2]);
}

torch::Tensor load_tensor(const string& tensor_path)
{
    torch::Tensor tensor;
    // cout << "Ready to load tensor" << endl;
    torch::load(tensor, tensor_path);
    return tensor;
}

vector<zkFC> load_model(const string& model_path, vector<Commitment>& generators)
{

    vector<zkFC> fcs;
    torch::jit::script::Module m;
    try {
        m = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        exit(-1);
    }
    uint parameter_count = 0;

    for (int i = 0; ; ++i) {
        if (!m.hasattr(to_string(i))) break;
        else if (!m.attr(to_string(i)).toModule().hasattr("weight")) {
            continue;
        }

        else{
            // Access weights of the i-th Linear layer
            auto linear_weight = m.attr(to_string(i)).toModule().attr("weight");

            // Get the weight on the GPU
            auto weight_tensor = linear_weight.toTensor().t();
            // cout << weight_tensor << endl;
            auto weight_shape = weight_tensor.sizes();
            int in_dim = weight_shape[0];
            int out_dim = weight_shape[1];
            parameter_count += (in_dim * out_dim);
            float* weight_ptr = weight_tensor.contiguous().data_ptr<float>();
            if (!weight_tensor.is_cuda()) throw std::runtime_error("Weight tensor is not on GPU");

            generators.push_back({1U<<((ceilLog2(in_dim * out_dim)+1)/2), G1Jacobian_generator});
            generators[generators.size()-1] *= FrTensor::random(generators[generators.size()-1].size);
            // cout << "Size of generators[" << generators.size()-1 << "] = " << generators[generators.size()-1].size << endl;

            fcs.push_back(zkFC::from_float_gpu_ptr(in_dim, out_dim, weight_ptr, generators[generators.size()-1]));
            // cout << "Size of fcs[" << fcs.size()-1 << "] = " << fcs[generators.size()-1].inputSize << "," << fcs[generators.size()-1].outputSize << endl;
            if (fcs.size () > 1 && fcs[fcs.size() - 2].outputSize != fcs[fcs.size() - 1].inputSize) {
                throw std::runtime_error("Incompatible layer sizes");
            }
        }
    }
    cout << "Total number of parameters: " << parameter_count << endl;
    
    return fcs;
}

// const uint NUM_BITS = 16;

int main(int argc, char *argv[]) // batch_size input_dim, hidden_dim, hidden_dim, ..., output_dim
{   
    vector<Commitment> generators;
    vector<zkFC> fcs = load_model(argv[1], generators);
    vector<zkReLU> relus (fcs.size () - 1);
    vector<FrTensor> Z_vec, A_vec;

    // Load the saved tensors
    torch::Tensor sample_input = load_tensor(argv[2]);
    int batch_size = sample_input.size(0);
    int input_dim = sample_input.size(1);
    

    // Get the pointer to a piece of contiguous memory for sample_input
    float* input_ptr = sample_input.contiguous().data_ptr<float>();
    if (!sample_input.is_cuda()) throw std::runtime_error("Sample input tensor is not on GPU");
    auto X = zkFC::load_float_gpu_input(batch_size, input_dim, input_ptr);
    // std::cout << "Sample Input: " << X << std::endl;

    // Run the inference
    auto Y_hat = fcnn_inference(X.mont(), fcs, relus, Z_vec, A_vec).unmont();
    ofstream outfile("demo.out");
    outfile << Y_hat << endl;
    outfile.close();

    Timer timer;
    timer.start();
    auto num_layer = fcs.size();
    // cout << "Running proof on layer "<< num_layer - 1 << "..." << endl;
    fcs[num_layer - 1].prove(A_vec[num_layer - 2], Y_hat, generators[num_layer - 1]);
    

    for(int i = num_layer - 2; i >= 0; -- i)
    {   
        // cout << "Running proof on layer "<< i << "..." << endl;
        relus[i].prove(Z_vec[i], A_vec[i]);
        FrTensor& A_ = (i > 0)? A_vec[i-1] : X;
        fcs[i].prove(A_, Z_vec[i], generators[i]);
    }
    timer.stop();

    cout << "Proof time: " << timer.getTotalTime() / batch_size << " seconds per data point." << endl;
    cout << "Current CUDA status: " << cudaGetLastError() << endl;
	return 0;
}