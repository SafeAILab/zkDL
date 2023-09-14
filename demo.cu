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

vector<Fr_t> random_vec(uint len)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned int> dist(0, UINT_MAX);
    vector<Fr_t> out(len);
    for (uint i = 0; i < len; ++ i) out[i] = {dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), dist(mt), 0};
    return out;
}

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

const int NUM_BITS = 14;

int main(int argc, char *argv[])
{
	// uint size = stoi(argv[1]);
	// uint window_size = stoi(argv[2]);
	// bool need_print = false;
	// if (argc > 3) need_print = stoi(argv[3]);

	uint num_layer = stoi(argv[1]);
    uint log_batch_size = stoi(argv[2]);
    uint batch_size = 1U << log_batch_size;
    uint log_width = stoi(argv[3]);
    uint width = 1U << log_width;

    vector<zkFC> fcs;
    vector<zkReLU> relus(num_layer - 1);

    for (uint i = 0; i < num_layer; ++ i) 
    {
        fcs.push_back({width, width, NUM_BITS});
    }

    auto X = FrTensor::random_int(batch_size * width, NUM_BITS);
    vector<FrTensor> Z_vec, A_vec;
    fcnn_inference(X, fcs, relus, Z_vec, A_vec);

    cout << "Current CUDA status: " << cudaGetLastError() << endl;
	return 0;
}