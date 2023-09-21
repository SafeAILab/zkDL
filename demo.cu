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

const uint NUM_BITS = 14;

int main(int argc, char *argv[])
{
    
	uint num_layer = stoi(argv[1]);
    uint log_batch_size = stoi(argv[2]);
    uint batch_size = 1U << log_batch_size;
    uint log_width = stoi(argv[3]);
    uint width = 1U << log_width;

    Commitment generators(width, G1Jacobian_generator);
    generators *= FrTensor::random(width);

    vector<zkFC> fcs;
    vector<zkReLU> relus(num_layer - 1);

    for (uint i = 0; i < num_layer; ++ i) 
    {
        fcs.push_back({width, width, NUM_BITS, generators});
    }

    auto X = FrTensor::random_int(batch_size * width, NUM_BITS);
    vector<FrTensor> Z_vec, A_vec;
    auto Y_hat = fcnn_inference(X.mont(), fcs, relus, Z_vec, A_vec).unmont();

    cout << "Running proof on layer "<< num_layer - 1 << "..." << endl;
    fcs[num_layer - 1].prove(A_vec[num_layer - 2], Y_hat, generators);
    

    for(int i = num_layer - 2; i >= 0; -- i)
    {   
        cout << "Running proof on layer "<< i << "..." << endl;
        relus[i].prove(Z_vec[i], A_vec[i]);
        FrTensor& A_ = (i > 0)? A_vec[i-1] : X;
        fcs[i].prove(A_, Z_vec[i], generators);
    }

    cout << "Current CUDA status: " << cudaGetLastError() << endl;
	return 0;
}