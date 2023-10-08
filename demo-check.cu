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

const uint NUM_BITS = 16;

int main(int argc, char *argv[]) // batch_size input_dim, hidden_dim, hidden_dim, ..., output_dim
{
	Commitment generators(2, G1Jacobian_generator);
    generators *= FrTensor::random(generators.size);

    auto weight1 = FrTensor::random_int(4, NUM_BITS);
    cout << weight1 << endl;
    zkFC fc1(2, 2, weight1.mont(), generators);
    zkReLU relu;
    auto weight2 = FrTensor::random_int(4, NUM_BITS);
    cout << weight2 << endl;
    zkFC fc2(2, 2, weight2.mont(), generators);
    
    auto X = FrTensor::random_int(2, NUM_BITS);
    cout << X << endl;

    auto Z1 = fc1(X.mont());
    // cout << Z1.unmont() << endl;
    auto A1 = relu(Z1);
    // cout << A1.unmont() << endl;
    auto Z2 = fc2(A1);
    // cout << Z2.unmont() << endl;

    cout << Z1.unmont() << endl;
    cout << A1.unmont() << endl;
    cout << Z2.unmont() << endl;

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