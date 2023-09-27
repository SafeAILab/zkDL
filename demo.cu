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

int main(int argc, char *argv[]) // batch_size input_dim, hidden_dim, hidden_dim, ..., output_dim
{
	uint num_layer = argc - 3;
    uint batch_size = stoi(argv[1]);
    uint log_batch_size = ceilLog2(batch_size);
    
    // uint log_width = stoi(argv[3]);
    // uint width = 1U << log_width;

    auto layer_dims = argv + 2;


    // uint max_log_dim = 0;
    // for (uint i = 0; i < num_layer; ++ i)
    // {   
    //     uint cur_log_dim = (ceilLog2(layer_dims[i]) + ceilLog2(layer_dims[i+1]) + 1) / 2;
    //     max_log_dim = max(max_log_dim, cur_log_dim);
    // }

    Commitment generators(1<<max_log_dim, G1Jacobian_generator);
    generators *= FrTensor::random(width);

    vector<zkFC> fcs;
    vector<zkReLU> relus(num_layer - 1);

    for (uint i = 0; i < num_layer; ++ i) 
    {   
        uint dim_in = stoi(layer_dims[i]);
        uint dim_out = stoi(layer_dims[i+1]);

        uint log_dim_in = ceilLog2(dim_in);
        uint log_dim_out = ceilLog2(dim_out);

        Commitment generators((log_dim_in+log_dim_out+1)/2, G1Jacobian_generator);
        generators *= FrTensor::random(generators.size);
        fcs.push_back(zkFC::random_fc(1<<log_dim_in, 1<<log_dim_out, NUM_BITS, generators));
    }

    auto X = FrTensor::random_int((1<<log_batch_size) * (1<<ceilLog2(stoi(layer_dims[0]))), NUM_BITS);
    vector<FrTensor> Z_vec, A_vec;
    auto Y_hat = fcnn_inference(X.mont(), fcs, relus, Z_vec, A_vec).unmont();

    // Timer timer;
    // timer.start();
    cout << "Running proof on layer "<< num_layer - 1 << "..." << endl;
    fcs[num_layer - 1].prove(A_vec[num_layer - 2], Y_hat, generators);
    

    for(int i = num_layer - 2; i >= 0; -- i)
    {   
        cout << "Running proof on layer "<< i << "..." << endl;
        relus[i].prove(Z_vec[i], A_vec[i]);
        FrTensor& A_ = (i > 0)? A_vec[i-1] : X;
        fcs[i].prove(A_, Z_vec[i], generators);
    }
    timer.stop();

    // cout << "Proof time: " << timer.getTotalTime() / batch_size << " seconds per data point." << endl;
    cout << "Current CUDA status: " << cudaGetLastError() << endl;
	return 0;
}